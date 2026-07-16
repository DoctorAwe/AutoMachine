"""Causal synchronous perception-to-response processor.

Raw observations are compressed to a fixed internal rate.  At each internal
step, new perception enters the first token layer while every deeper layer
receives the *previous-step* state of its predecessor.  A readout over current
perception and all layers produces the synchronous response.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SynchronousControlConfig:
    image_size: int = 96
    input_channels: int = 3
    auxiliary_features: int = 0
    response_dim: int = 16
    frames_per_step: int = 1
    token_dim: int = 64
    spatial_grid: int = 4
    state_tokens: tuple[int, ...] = (16, 16, 24, 32)
    num_heads: int = 4
    mlp_ratio: int = 2
    dropout: float = 0.0
    response_activation: str = "none"

    def __post_init__(self) -> None:
        positive = (
            self.image_size, self.input_channels, self.response_dim,
            self.frames_per_step, self.token_dim, self.spatial_grid,
            self.num_heads, self.mlp_ratio,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Model dimensions and frames_per_step must be positive")
        if self.auxiliary_features < 0:
            raise ValueError("auxiliary_features cannot be negative")
        if self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        if not self.state_tokens or any(count <= 0 for count in self.state_tokens):
            raise ValueError("state_tokens must contain positive values")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.response_activation not in {"none", "tanh"}:
            raise ValueError("response_activation must be 'none' or 'tanh'")


@dataclass
class ControlState:
    """Token pipeline state plus raw samples awaiting a complete input group."""

    layers: tuple[Tensor, ...]
    step: int = 0
    pending_video: Tensor | None = None
    pending_auxiliary: Tensor | None = None

    def detach(self) -> "ControlState":
        return ControlState(
            tuple(layer.detach() for layer in self.layers),
            self.step,
            None if self.pending_video is None else self.pending_video.detach(),
            None if self.pending_auxiliary is None else self.pending_auxiliary.detach(),
        )


class FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * ratio, dim), nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class TokenResampler(nn.Module):
    """Convert a token layer to the capacity expected by the next layer."""

    def __init__(self, target_tokens: int, config: SynchronousControlConfig) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, target_tokens, config.token_dim))
        self.norm = nn.LayerNorm(config.token_dim)
        self.attention = nn.MultiheadAttention(
            config.token_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        query = self.query.expand(tokens.shape[0], -1, -1)
        normalized = self.norm(tokens)
        result, _ = self.attention(query, normalized, normalized, need_weights=False)
        return result


class CausalVisualEncoder(nn.Module):
    """Compress K raw frames to one fixed-size spatial-token step."""

    def __init__(self, config: SynchronousControlConfig) -> None:
        super().__init__()
        dim = config.token_dim
        self.config = config
        self.backbone = nn.Sequential(
            nn.Conv2d(config.input_channels, dim // 4, 5, stride=2, padding=2),
            nn.GroupNorm(4, dim // 4), nn.GELU(),
            nn.Conv2d(dim // 4, dim // 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, dim // 2), nn.GELU(),
            nn.Conv2d(dim // 2, dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, dim), nn.GELU(),
        )
        token_count = config.spatial_grid**2
        self.spatial_position = nn.Parameter(torch.empty(1, 1, token_count, dim))
        self.group_position = nn.Parameter(
            torch.empty(1, config.frames_per_step, 1, dim)
        )
        self.group_query = nn.Parameter(torch.empty(1, token_count, dim))
        self.group_attention = nn.MultiheadAttention(
            dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(dim)
        for parameter in (self.spatial_position, self.group_position, self.group_query):
            nn.init.trunc_normal_(parameter, std=0.02)

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected video [B,T,C,H,W], got {tuple(video.shape)}")
        batch, raw_time, channels, height, width = video.shape
        group = self.config.frames_per_step
        if channels != self.config.input_channels:
            raise ValueError(f"Expected {self.config.input_channels} channels, got {channels}")
        if raw_time == 0 or raw_time % group:
            raise ValueError("Encoder input must contain complete frame groups")

        features = self.backbone(video.reshape(batch * raw_time, channels, height, width))
        grid = self.config.spatial_grid
        features = F.adaptive_avg_pool2d(features, (grid, grid))
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens.reshape(batch, raw_time, grid * grid, -1) + self.spatial_position
        internal_time = raw_time // group
        tokens = tokens.reshape(batch, internal_time, group, grid * grid, -1)
        tokens = tokens + self.group_position.unsqueeze(1)
        source = tokens.flatten(2, 3).reshape(batch * internal_time, group * grid * grid, -1)
        query = self.group_query.expand(batch * internal_time, -1, -1)
        encoded, _ = self.group_attention(query, source, source, need_weights=False)
        encoded = self.output_norm(encoded)
        return encoded.reshape(batch, internal_time, grid * grid, -1)


class PerceptionFusion(nn.Module):
    """Append optional synchronous vector sensors as an additional token."""

    def __init__(self, config: SynchronousControlConfig) -> None:
        super().__init__()
        self.config = config
        if config.auxiliary_features:
            self.auxiliary_encoder = nn.Sequential(
                nn.LayerNorm(config.auxiliary_features),
                nn.Linear(config.auxiliary_features, config.token_dim),
                nn.GELU(), nn.Linear(config.token_dim, config.token_dim),
                nn.LayerNorm(config.token_dim),
            )

    def forward(self, visual: Tensor, auxiliary: Tensor | None) -> Tensor:
        if not self.config.auxiliary_features:
            if auxiliary is not None:
                raise ValueError("Model was configured without auxiliary features")
            return visual
        if auxiliary is None:
            raise ValueError("Auxiliary sensor input is required")
        if auxiliary.shape[:2] != visual.shape[:2] or auxiliary.shape[-1] != self.config.auxiliary_features:
            raise ValueError(
                f"Expected grouped auxiliary [B,T,{self.config.auxiliary_features}], "
                f"got {tuple(auxiliary.shape)}"
            )
        token = self.auxiliary_encoder(auxiliary).unsqueeze(2)
        return torch.cat([visual, token], dim=2)


class TokenLayer(nn.Module):
    """Fuse incoming tokens with the current layer, then self-attend."""

    def __init__(self, config: SynchronousControlConfig) -> None:
        super().__init__()
        dim = config.token_dim
        self.state_norm = nn.LayerNorm(dim)
        self.input_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, config.mlp_ratio, config.dropout)
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, state: Tensor, incoming: Tensor) -> Tensor:
        update, _ = self.cross_attention(
            self.state_norm(state), self.input_norm(incoming), self.input_norm(incoming),
            need_weights=False,
        )
        candidate = state + update
        normalized = self.self_norm(candidate)
        context, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        candidate = candidate + context
        candidate = candidate + self.ffn(self.ffn_norm(candidate))
        gate = torch.sigmoid(self.gate(torch.cat([state, candidate], dim=-1)))
        return gate * candidate + (1.0 - gate) * state


class TokenPipelineMemory(nn.Module):
    """Temporal shift pipeline: old layer i moves into layer i+1."""

    def __init__(self, config: SynchronousControlConfig) -> None:
        super().__init__()
        self.config = config
        self.initial_states = nn.ParameterList(
            [nn.Parameter(torch.empty(1, count, config.token_dim)) for count in config.state_tokens]
        )
        self.layers = nn.ModuleList([TokenLayer(config) for _ in config.state_tokens])
        self.transfers = nn.ModuleList(
            [TokenResampler(config.state_tokens[index], config) for index in range(1, len(config.state_tokens))]
        )
        self.readout_query = nn.Parameter(torch.empty(1, 1, config.token_dim))
        self.readout_attention = nn.MultiheadAttention(
            config.token_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.readout_norm = nn.LayerNorm(config.token_dim)
        for state in self.initial_states:
            nn.init.trunc_normal_(state, std=0.02)
        nn.init.trunc_normal_(self.readout_query, std=0.02)

    def init_state(self, batch: int, device: torch.device) -> ControlState:
        return ControlState(
            tuple(state.expand(batch, -1, -1).clone().to(device) for state in self.initial_states)
        )

    def forward_step(self, perception: Tensor, state: ControlState) -> tuple[Tensor, ControlState]:
        if len(state.layers) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} token layers, got {len(state.layers)}")
        # Every incoming value is derived from the same previous-step snapshot.
        incoming = [perception]
        incoming.extend(
            transfer(state.layers[index]) for index, transfer in enumerate(self.transfers)
        )
        new_layers = tuple(
            layer(old_layer, layer_input)
            for layer, old_layer, layer_input in zip(self.layers, state.layers, incoming)
        )
        bank = torch.cat([perception, *new_layers], dim=1)
        query = self.readout_query.expand(perception.shape[0], -1, -1)
        readout, _ = self.readout_attention(query, bank, bank, need_weights=False)
        return self.readout_norm(readout[:, 0]), ControlState(new_layers, state.step + 1)


class ResponseDecoder(nn.Module):
    def __init__(self, config: SynchronousControlConfig) -> None:
        super().__init__()
        self.activation = config.response_activation
        self.net = nn.Sequential(
            nn.LayerNorm(config.token_dim),
            nn.Linear(config.token_dim, config.token_dim * 2), nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.token_dim * 2, config.response_dim),
        )

    def forward(self, token: Tensor) -> Tensor:
        response = self.net(token)
        return torch.tanh(response) if self.activation == "tanh" else response


class SynchronousControlProcessor(nn.Module):
    """Fixed-rate causal processor for synchronous response/control signals."""

    def __init__(self, config: SynchronousControlConfig | None = None) -> None:
        super().__init__()
        self.config = config or SynchronousControlConfig()
        self.encoder = CausalVisualEncoder(self.config)
        self.fusion = PerceptionFusion(self.config)
        self.memory = TokenPipelineMemory(self.config)
        self.decoder = ResponseDecoder(self.config)

    def _join_pending(
        self, video: Tensor, auxiliary: Tensor | None, state: ControlState
    ) -> tuple[Tensor, Tensor | None]:
        if state.pending_video is not None:
            video = torch.cat([state.pending_video, video], dim=1)
        if self.config.auxiliary_features:
            if auxiliary is None:
                raise ValueError("Auxiliary sensor input is required")
            if state.pending_auxiliary is not None:
                auxiliary = torch.cat([state.pending_auxiliary, auxiliary], dim=1)
        return video, auxiliary

    def forward_chunk(
        self,
        video: Tensor,
        auxiliary: Tensor | None = None,
        state: ControlState | None = None,
    ) -> tuple[Tensor, ControlState]:
        if video.ndim != 5 or video.shape[1] == 0:
            raise ValueError("video must be non-empty [B,T,C,H,W]")
        if state is None:
            state = self.memory.init_state(video.shape[0], video.device)
        video, auxiliary = self._join_pending(video, auxiliary, state)
        group = self.config.frames_per_step
        complete = (video.shape[1] // group) * group
        pending_video = video[:, complete:]
        pending_auxiliary = None if auxiliary is None else auxiliary[:, complete:]

        if complete == 0:
            empty = video.new_empty((video.shape[0], 0, self.config.response_dim))
            return empty, ControlState(state.layers, state.step, pending_video, pending_auxiliary)

        grouped_video = video[:, :complete]
        visual = self.encoder(grouped_video)
        grouped_auxiliary = None
        if auxiliary is not None:
            raw_auxiliary = auxiliary[:, :complete]
            grouped_auxiliary = raw_auxiliary.reshape(
                raw_auxiliary.shape[0], -1, group, raw_auxiliary.shape[-1]
            ).mean(dim=2)
        perceptions = self.fusion(visual, grouped_auxiliary)

        outputs: list[Tensor] = []
        running = ControlState(state.layers, state.step)
        for timestep in range(perceptions.shape[1]):
            readout, running = self.memory.forward_step(perceptions[:, timestep], running)
            outputs.append(self.decoder(readout))
        next_state = ControlState(
            running.layers, running.step, pending_video, pending_auxiliary
        )
        return torch.stack(outputs, dim=1), next_state

    def forward(self, video: Tensor, auxiliary: Tensor | None = None) -> Tensor:
        responses, state = self.forward_chunk(video, auxiliary)
        if state.pending_video is not None and state.pending_video.shape[1]:
            raise ValueError("Full forward input length must be divisible by frames_per_step")
        return responses
