"""Small hierarchical stream-processing model.

This module implements the first executable shape of the "neural processor"
idea from AGENTS.md:

- encode a short chunk of raw video frames into compact tokens,
- maintain multiple token-state layers with different capacities,
- fuse each incoming step into persistent state by attention + gated update,
- decode the current readout token grid back into a video frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NeuralStreamProcessorConfig:
    """Configuration for the small video next-frame prototype."""

    image_size: int = 64
    input_channels: int = 3
    chunk_size: int = 3
    token_dim: int = 128
    encoder_grid: int = 8
    state_tokens: tuple[int, ...] = (64, 64, 96, 128, 192)
    num_heads: int = 4
    mlp_ratio: int = 4
    output_channels: int = 3

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.chunk_size <= 0 or self.encoder_grid <= 0:
            raise ValueError("image_size, chunk_size and encoder_grid must be positive")
        if self.token_dim <= 0 or self.num_heads <= 0 or self.token_dim % self.num_heads:
            raise ValueError("token_dim must be positive and divisible by num_heads")
        if not self.state_tokens or any(count <= 0 for count in self.state_tokens):
            raise ValueError("state_tokens must contain positive token counts")

    @property
    def encoder_tokens(self) -> int:
        return self.encoder_grid * self.encoder_grid


class StreamEncoder(nn.Module):
    """Compresses a short video chunk into a fixed token grid."""

    def __init__(self, config: NeuralStreamProcessorConfig) -> None:
        super().__init__()
        dim = config.token_dim
        self.config = config
        # Keep time as an explicit dimension.  The former implementation folded
        # frames into channels, which made frame order unnecessarily hard to learn.
        self.net = nn.Sequential(
            nn.Conv3d(
                config.input_channels,
                dim // 2,
                kernel_size=(3, 5, 5),
                stride=(1, 2, 2),
                padding=(1, 2, 2),
            ),
            nn.GELU(),
            nn.Conv3d(
                dim // 2,
                dim,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=1,
            ),
            nn.GELU(),
            nn.Conv3d(
                dim,
                dim,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=1,
            ),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.temporal_score = nn.Conv3d(dim, 1, kernel_size=1)
        self.pos = nn.Parameter(torch.zeros(1, config.encoder_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, chunk: Tensor) -> Tensor:
        """Encode a frame chunk.

        Args:
            chunk: Tensor shaped [batch, chunk_size, channels, height, width].

        Returns:
            Tensor shaped [batch, encoder_grid * encoder_grid, token_dim].
        """
        bsz, steps, channels, height, width = chunk.shape
        expected = self.config
        if steps != expected.chunk_size or channels != expected.input_channels:
            raise ValueError(
                f"Expected chunk [B, {expected.chunk_size}, "
                f"{expected.input_channels}, H, W], got {tuple(chunk.shape)}"
            )

        x = self.net(chunk.transpose(1, 2))
        # Content-dependent causal-chunk pooling preserves ordering information
        # while still producing a fixed-rate token grid.
        weights = self.temporal_score(x).mean(dim=(-1, -2), keepdim=True).softmax(dim=2)
        x = self.proj((x * weights).sum(dim=2))
        if x.shape[-2:] != (expected.encoder_grid, expected.encoder_grid):
            x = F.adaptive_avg_pool2d(x, (expected.encoder_grid, expected.encoder_grid))
        x = x.flatten(2).transpose(1, 2)
        return x + self.pos


class TokenResampler(nn.Module):
    """Maps an arbitrary token sequence to a target token count."""

    def __init__(self, target_tokens: int, token_dim: int, num_heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, target_tokens, token_dim))
        self.norm = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        bsz = tokens.shape[0]
        query = self.query.expand(bsz, -1, -1)
        key_value = self.norm(tokens)
        resampled, _ = self.attn(query, key_value, key_value, need_weights=False)
        return resampled


class FeedForward(nn.Module):
    def __init__(self, token_dim: int, mlp_ratio: int) -> None:
        super().__init__()
        hidden = token_dim * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, token_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class StateFusionBlock(nn.Module):
    """Attention + gate update for one persistent token-state layer."""

    def __init__(
        self,
        state_tokens: int,
        token_dim: int,
        num_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.state_tokens = state_tokens
        self.resampler = TokenResampler(state_tokens, token_dim, num_heads)
        self.state_norm = nn.LayerNorm(token_dim)
        self.incoming_norm = nn.LayerNorm(token_dim)
        self.cross_attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(token_dim)
        self.self_attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = FeedForward(token_dim, mlp_ratio)
        self.gate = nn.Linear(token_dim * 2, token_dim)

    def forward(self, state: Tensor, incoming: Tensor) -> Tensor:
        incoming = self.resampler(incoming)
        state_norm = self.state_norm(state)
        incoming_norm = self.incoming_norm(incoming)

        candidate, _ = self.cross_attn(
            query=state_norm,
            key=incoming_norm,
            value=incoming_norm,
            need_weights=False,
        )
        candidate = state + candidate

        self_context, _ = self.self_attn(
            query=self.self_norm(candidate),
            key=self.self_norm(candidate),
            value=self.self_norm(candidate),
            need_weights=False,
        )
        candidate = candidate + self_context
        candidate = candidate + self.ffn(self.ffn_norm(candidate))

        gate = torch.sigmoid(self.gate(torch.cat([state, candidate], dim=-1)))
        return gate * candidate + (1.0 - gate) * state


class HierarchicalMemoryPipeline(nn.Module):
    """Persistent state machine with variable-capacity token layers."""

    def __init__(self, config: NeuralStreamProcessorConfig) -> None:
        super().__init__()
        self.config = config
        self.initial_states = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, count, config.token_dim))
                for count in config.state_tokens
            ]
        )
        self.layers = nn.ModuleList(
            [
                StateFusionBlock(
                    state_tokens=count,
                    token_dim=config.token_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                )
                for count in config.state_tokens
            ]
        )
        # Read from every memory scale. Shallow layers retain recent detail while
        # larger/deeper layers provide a slower, higher-capacity context.
        self.readout_query = nn.Parameter(
            torch.empty(1, config.encoder_tokens, config.token_dim)
        )
        self.readout_norms = nn.ModuleList(
            [nn.LayerNorm(config.token_dim) for _ in config.state_tokens]
        )
        self.readout_attn = nn.MultiheadAttention(
            config.token_dim, config.num_heads, batch_first=True
        )
        self.readout_output_norm = nn.LayerNorm(config.token_dim)
        nn.init.trunc_normal_(self.readout_query, std=0.02)

    def init_state(self, batch_size: int, device: torch.device | None = None) -> list[Tensor]:
        """Create a fresh recurrent state list for a batch."""
        states: list[Tensor] = []
        for initial in self.initial_states:
            base = initial if device is None else initial.to(device)
            states.append(base.expand(batch_size, -1, -1).clone())
        return states

    def forward(
        self,
        encoded: Tensor,
        states: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Advance the memory pipeline by one encoded time step."""
        if states is None:
            states = self.init_state(encoded.shape[0], encoded.device)
        if len(states) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} states, got {len(states)}")

        incoming = encoded
        new_states: list[Tensor] = []
        for layer, state in zip(self.layers, states):
            updated = layer(state, incoming)
            new_states.append(updated)
            incoming = updated
        memory_bank = torch.cat(
            [norm(state) for norm, state in zip(self.readout_norms, new_states)], dim=1
        )
        query = self.readout_query.expand(encoded.shape[0], -1, -1)
        readout, _ = self.readout_attn(
            query=query, key=memory_bank, value=memory_bank, need_weights=False
        )
        return self.readout_output_norm(readout), new_states


class StreamDecoder(nn.Module):
    """Decodes a token grid back to one output frame."""

    def __init__(self, config: NeuralStreamProcessorConfig) -> None:
        super().__init__()
        self.config = config
        self.readout = TokenResampler(config.encoder_tokens, config.token_dim, config.num_heads)
        dim = config.token_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(dim // 4, config.output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = self.readout(tokens)
        bsz = tokens.shape[0]
        grid = self.config.encoder_grid
        x = tokens.transpose(1, 2).reshape(bsz, self.config.token_dim, grid, grid)
        frame = self.net(x)
        if frame.shape[-1] != self.config.image_size:
            frame = F.interpolate(
                frame,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return frame


class NeuralStreamProcessor(nn.Module):
    """End-to-end small stream processor for video next-frame prediction."""

    def __init__(self, config: NeuralStreamProcessorConfig | None = None) -> None:
        super().__init__()
        self.config = config or NeuralStreamProcessorConfig()
        self.encoder = StreamEncoder(self.config)
        self.memory = HierarchicalMemoryPipeline(self.config)
        self.decoder = StreamDecoder(self.config)

    def forward_step(
        self,
        chunk: Tensor,
        states: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Process one chunk and predict one output frame."""
        encoded = self.encoder(chunk)
        readout, new_states = self.memory(encoded, states)
        frame = self.decoder(readout)
        return frame, new_states

    def forward(self, video: Tensor) -> Tensor:
        """Process a whole video sequence chunk-by-chunk.

        Args:
            video: Tensor shaped [batch, time, channels, height, width].

        Returns:
            Tensor shaped [batch, output_steps, channels, height, width].
        """
        if video.ndim != 5:
            raise ValueError(f"Expected [B, T, C, H, W], got {tuple(video.shape)}")
        chunk = self.config.chunk_size
        if video.shape[1] < chunk:
            raise ValueError(f"Video needs at least {chunk} frames")

        states: list[Tensor] | None = None
        outputs: list[Tensor] = []
        for start in range(0, video.shape[1] - chunk + 1):
            frame, states = self.forward_step(video[:, start : start + chunk], states)
            outputs.append(frame)
        return torch.stack(outputs, dim=1)
