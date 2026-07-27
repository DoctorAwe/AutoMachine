"""Causal synchronous perception-to-response processor.

Raw observations are compressed to a fixed internal rate.  At each internal
step, new perception enters the first token layer while every deeper layer
receives the *previous-step* state of its predecessor.  A readout over current
perception and all layers produces the synchronous response.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# Sixteen causal stages give the pipeline sixteen encoded-time steps of direct
# propagation history.  The tail grows wider to provide higher-capacity,
# slower-changing state without making every early stage expensive.
DEFAULT_STATE_TOKENS = (
    16, 16, 16, 16, 16, 16, 16, 16,
    24, 24, 24, 24,
    32, 32,
    48, 64,
)


@dataclass(frozen=True)
class SynchronousControlConfig:
    image_size: int = 96
    input_channels: int = 3
    auxiliary_features: int = 0
    response_dim: int = 16
    frames_per_step: int = 1
    token_dim: int = 64
    spatial_grid: int = 4
    state_tokens: tuple[int, ...] = DEFAULT_STATE_TOKENS
    num_heads: int = 4
    mlp_ratio: int = 2
    dropout: float = 0.0
    response_activation: str = "none"
    fusion_acceptance_start: float = 0.95
    fusion_acceptance_end: float = 0.10
    fusion_threshold_start: float = 0.05
    fusion_threshold_end: float = 0.20
    fusion_temperature: float = 0.025
    associative_memory_enabled: bool = True
    associative_memory_capacity: int = 256
    associative_value_tokens: int = 8
    associative_top_k: int = 4
    associative_retrieval_threshold: float = 0.72
    associative_retrieval_temperature: float = 0.10
    associative_merge_key_threshold: float = 0.88
    associative_merge_value_threshold: float = 0.80
    associative_write_threshold: float = 0.65
    associative_replacement_margin: float = 0.10
    associative_protected_fraction: float = 0.125
    associative_candidate_fraction: float = 0.25

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
        if not 0.0 < self.fusion_acceptance_start < 1.0:
            raise ValueError("fusion_acceptance_start must be between 0 and 1")
        if not 0.0 < self.fusion_acceptance_end < 1.0:
            raise ValueError("fusion_acceptance_end must be between 0 and 1")
        if self.fusion_acceptance_start < self.fusion_acceptance_end:
            raise ValueError("fusion acceptance must not increase with pipeline depth")
        if self.fusion_threshold_start < 0.0 or self.fusion_threshold_end < 0.0:
            raise ValueError("fusion thresholds cannot be negative")
        if self.fusion_threshold_start > self.fusion_threshold_end:
            raise ValueError("fusion threshold must not decrease with pipeline depth")
        if self.fusion_temperature <= 0.0:
            raise ValueError("fusion_temperature must be positive")
        if self.associative_memory_capacity <= 0:
            raise ValueError("associative_memory_capacity must be positive")
        if not 0 < self.associative_value_tokens:
            raise ValueError("associative_value_tokens must be positive")
        if not 0 < self.associative_top_k <= self.associative_memory_capacity:
            raise ValueError("associative_top_k must be within memory capacity")
        probabilities = (
            self.associative_retrieval_threshold,
            self.associative_merge_key_threshold,
            self.associative_merge_value_threshold,
            self.associative_write_threshold,
            self.associative_protected_fraction,
            self.associative_candidate_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("Associative thresholds/fractions must be in [0, 1]")
        if self.associative_protected_fraction + self.associative_candidate_fraction >= 1.0:
            raise ValueError("Protected and candidate fractions must leave stable capacity")
        if self.associative_replacement_margin < 0:
            raise ValueError("associative_replacement_margin cannot be negative")
        if self.associative_retrieval_temperature <= 0:
            raise ValueError("associative_retrieval_temperature must be positive")


@dataclass
class AssociativeMemoryState:
    """Fixed-capacity, persistent episodic memory carried between chunks.

    Tier 0 is a replaceable candidate, tier 1 is stable, and tier 2 is
    protected. Contents are state, not trainable parameters.
    """

    keys: Tensor
    values: Tensor
    importance: Tensor
    confidence: Tensor
    usage: Tensor
    age: Tensor
    occupied: Tensor
    tier: Tensor

    def detach(self) -> "AssociativeMemoryState":
        return AssociativeMemoryState(
            self.keys.detach(), self.values.detach(), self.importance.detach(),
            self.confidence.detach(), self.usage.detach(), self.age.detach(),
            self.occupied.detach(), self.tier.detach(),
        )

    def to(self, device: torch.device) -> "AssociativeMemoryState":
        return AssociativeMemoryState(
            self.keys.to(device), self.values.to(device), self.importance.to(device),
            self.confidence.to(device), self.usage.to(device), self.age.to(device),
            self.occupied.to(device), self.tier.to(device),
        )


@dataclass
class ControlState:
    """Token pipeline state plus raw samples awaiting a complete input group."""

    layers: tuple[Tensor, ...]
    step: int = 0
    pending_video: Tensor | None = None
    pending_auxiliary: Tensor | None = None
    associative: AssociativeMemoryState | None = None

    def detach(self) -> "ControlState":
        return ControlState(
            tuple(layer.detach() for layer in self.layers),
            self.step,
            None if self.pending_video is None else self.pending_video.detach(),
            None if self.pending_auxiliary is None else self.pending_auxiliary.detach(),
            None if self.associative is None else self.associative.detach(),
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

    def __init__(self, config: SynchronousControlConfig, depth_index: int, depth_count: int) -> None:
        super().__init__()
        dim = config.token_dim
        self.depth_fraction = depth_index / max(depth_count - 1, 1)
        self.initial_acceptance = (
            config.fusion_acceptance_start * (1.0 - self.depth_fraction)
            + config.fusion_acceptance_end * self.depth_fraction
        )
        self.fusion_threshold = (
            config.fusion_threshold_start * (1.0 - self.depth_fraction)
            + config.fusion_threshold_end * self.depth_fraction
        )
        self.fusion_temperature = config.fusion_temperature
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
        # The depth prior is only an initialization. The learned gate can move
        # away from it during training when the data supports another policy.
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(
            self.gate.bias,
            math.log(self.initial_acceptance / (1.0 - self.initial_acceptance)),
        )

    def fusion_gate(self, state: Tensor, candidate: Tensor, stimulus_update: Tensor) -> Tensor:
        learned_gate = torch.sigmoid(self.gate(torch.cat([state, candidate], dim=-1)))
        # Attention-update RMS is a learned salience measure. Weak proposals
        # close the gate; sufficiently strong proposals can use the layer's
        # depth-dependent acceptance capacity.
        update_strength = torch.sqrt(torch.mean(stimulus_update.float().square(), dim=-1, keepdim=True) + 1e-12)
        significance = torch.sigmoid(
            (update_strength - self.fusion_threshold) / self.fusion_temperature
        ).to(learned_gate.dtype)
        return learned_gate * significance

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
        gate = self.fusion_gate(state, candidate, update)
        return gate * candidate + (1.0 - gate) * state


class FixedAssociativeMemory(nn.Module):
    """Bounded episodic memory with differentiable retrieval and stateful writes."""

    def __init__(self, config: SynchronousControlConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.token_dim
        value_tokens = config.associative_value_tokens
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.value_query = nn.Parameter(torch.empty(1, value_tokens, dim))
        self.value_attention = nn.MultiheadAttention(
            dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.current_source = nn.Parameter(torch.empty(1, 1, dim))
        self.recalled_source = nn.Parameter(torch.empty(1, 1, dim))
        self.current_norm = nn.LayerNorm(dim)
        self.recall_norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.value_query, std=0.02)
        nn.init.trunc_normal_(self.current_source, std=0.02)
        nn.init.trunc_normal_(self.recalled_source, std=0.02)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> AssociativeMemoryState:
        capacity = self.config.associative_memory_capacity
        dim = self.config.token_dim
        value_tokens = self.config.associative_value_tokens
        zeros = torch.zeros
        return AssociativeMemoryState(
            keys=zeros(batch, capacity, dim, device=device, dtype=dtype),
            values=zeros(batch, capacity, value_tokens, dim, device=device, dtype=dtype),
            importance=zeros(batch, capacity, device=device),
            confidence=zeros(batch, capacity, device=device),
            usage=zeros(batch, capacity, device=device),
            age=zeros(batch, capacity, device=device),
            occupied=torch.zeros(batch, capacity, device=device, dtype=torch.bool),
            tier=torch.zeros(batch, capacity, device=device, dtype=torch.int8),
        )

    def encode(self, perception: Tensor) -> tuple[Tensor, Tensor]:
        pooled = perception.mean(dim=1)
        query = F.normalize(self.query_encoder(pooled), dim=-1)
        value_query = self.value_query.expand(perception.shape[0], -1, -1)
        value, _ = self.value_attention(value_query, perception, perception, need_weights=False)
        return query, value

    def current_tokens(self, value: Tensor) -> Tensor:
        """Trainable compression used now and stored for later association."""
        return self.current_norm(value + self.current_source)

    def retrieve(
        self, perception: Tensor, state: AssociativeMemoryState
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        query, proposed_value = self.encode(perception)
        similarities = torch.einsum("bd,bcd->bc", query, F.normalize(state.keys, dim=-1))
        similarities = similarities.masked_fill(~state.occupied, -1.0)
        reliability = (0.5 + 0.5 * state.confidence) * (
            0.9 + 0.1 * state.importance
        )
        scores = similarities * reliability
        top_scores, top_indices = torch.topk(
            scores, k=self.config.associative_top_k, dim=-1
        )
        valid = top_scores >= self.config.associative_retrieval_threshold
        safe_scores = top_scores.masked_fill(~valid, -1e4)
        weights = torch.softmax(
            safe_scores / self.config.associative_retrieval_temperature, dim=-1
        ) * valid.to(top_scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        batch_index = torch.arange(perception.shape[0], device=perception.device)[:, None]
        selected = state.values[batch_index, top_indices]
        recalled = torch.sum(weights[..., None, None] * selected, dim=1)
        recalled = self.recall_norm(recalled + self.recalled_source)
        recalled = recalled * valid.any(dim=-1)[:, None, None]
        return recalled, query, proposed_value, similarities

    @torch.no_grad()
    def write(
        self,
        state: AssociativeMemoryState,
        query: Tensor,
        proposed_value: Tensor,
        similarities: Tensor,
        external_salience: Tensor | None = None,
    ) -> AssociativeMemoryState:
        # Clone so prior states remain immutable snapshots for causal/debug use.
        result = AssociativeMemoryState(
            state.keys.clone(), state.values.clone(), state.importance.clone(),
            state.confidence.clone(), state.usage.clone(), state.age.clone(),
            state.occupied.clone(), state.tier.clone(),
        )
        result.age[result.occupied] += 1
        capacity = self.config.associative_memory_capacity
        candidate_start = int(capacity * (1.0 - self.config.associative_candidate_fraction))
        protected_limit = int(capacity * self.config.associative_protected_fraction)

        for batch in range(query.shape[0]):
            occupied = result.occupied[batch]
            best_similarity, best_index = similarities[batch].max(dim=0)
            novelty = 1.0 - best_similarity.clamp(0.0, 1.0) if occupied.any() else query.new_tensor(1.0)
            salience = novelty if external_salience is None else (
                0.5 * novelty + 0.5 * external_salience[batch].clamp(0.0, 1.0)
            )
            if salience < self.config.associative_write_threshold and (
                not occupied.any() or best_similarity < self.config.associative_merge_key_threshold
            ):
                continue

            merge = bool(occupied.any() and best_similarity >= self.config.associative_merge_key_threshold)
            if merge:
                old_value = result.values[batch, best_index]
                value_similarity = F.cosine_similarity(
                    old_value.flatten(), proposed_value[batch].flatten(), dim=0
                )
                merge = bool(value_similarity >= self.config.associative_merge_value_threshold)
            if merge:
                count = result.usage[batch, best_index]
                rate = 1.0 / (count + 2.0)
                result.keys[batch, best_index] = F.normalize(
                    (1.0 - rate) * result.keys[batch, best_index] + rate * query[batch], dim=-1
                )
                result.values[batch, best_index].lerp_(proposed_value[batch], rate)
                result.usage[batch, best_index] += 1
                result.confidence[batch, best_index] = (
                    0.9 * result.confidence[batch, best_index] + 0.1 * best_similarity
                ).clamp(0.0, 1.0)
                result.importance[batch, best_index] = torch.maximum(
                    result.importance[batch, best_index], salience
                )
                result.age[batch, best_index] = 0
                if result.usage[batch, best_index] >= 8:
                    result.tier[batch, best_index] = max(
                        int(result.tier[batch, best_index]), 1
                    )
                if result.usage[batch, best_index] >= 32 and result.importance[batch, best_index] >= 0.8:
                    result.tier[batch, best_index] = 2
                    protected_empty = torch.flatnonzero(
                        ~result.occupied[batch, :protected_limit]
                    )
                    if len(protected_empty) and int(best_index) >= protected_limit:
                        destination = protected_empty[0]
                        for tensor in (
                            result.keys, result.values, result.importance, result.confidence,
                            result.usage, result.age, result.occupied, result.tier,
                        ):
                            tensor[batch, destination] = tensor[batch, best_index]
                        result.keys[batch, best_index].zero_()
                        result.values[batch, best_index].zero_()
                        result.importance[batch, best_index] = 0
                        result.confidence[batch, best_index] = 0
                        result.usage[batch, best_index] = 0
                        result.age[batch, best_index] = 0
                        result.occupied[batch, best_index] = False
                        result.tier[batch, best_index] = 0
                continue

            slot_order = torch.cat((
                torch.arange(candidate_start, capacity, device=query.device),
                torch.arange(protected_limit, candidate_start, device=query.device),
            ))
            empty_candidates = slot_order[~occupied[slot_order]]
            if len(empty_candidates):
                slot = empty_candidates[0]
            else:
                replaceable = occupied & (result.tier[batch] < 2)
                # Protected physical reserve is only usable by promoted memories.
                replaceable[:protected_limit] = False
                if not replaceable.any():
                    continue
                recency = torch.exp(-result.age[batch] / 128.0)
                usage = torch.log1p(result.usage[batch]) / math.log(33.0)
                retention = (
                    0.35 * result.importance[batch]
                    + 0.25 * result.confidence[batch]
                    + 0.20 * usage.clamp(max=1.0)
                    + 0.20 * recency
                )
                retention = retention.masked_fill(~replaceable, float("inf"))
                slot = retention.argmin()
                if salience <= retention[slot] + self.config.associative_replacement_margin:
                    continue
            result.keys[batch, slot] = query[batch]
            result.values[batch, slot] = proposed_value[batch]
            result.importance[batch, slot] = salience
            result.confidence[batch, slot] = 0.5
            result.usage[batch, slot] = 1
            result.age[batch, slot] = 0
            result.occupied[batch, slot] = True
            result.tier[batch, slot] = 0
        return result


class TokenPipelineMemory(nn.Module):
    """Temporal shift pipeline: old layer i moves into layer i+1."""

    def __init__(self, config: SynchronousControlConfig) -> None:
        super().__init__()
        self.config = config
        self.initial_states = nn.ParameterList(
            [nn.Parameter(torch.empty(1, count, config.token_dim)) for count in config.state_tokens]
        )
        self.layers = nn.ModuleList(
            [
                TokenLayer(config, index, len(config.state_tokens))
                for index in range(len(config.state_tokens))
            ]
        )
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
        default_config = SynchronousControlConfig()
        legacy_config = config is not None and not hasattr(config, "associative_memory_enabled")
        if config is None:
            self.config = default_config
        else:
            # Dataclass instances stored in older checkpoints do not gain newly
            # added fields when unpickled. Rebuild them with current defaults.
            values = {
                name: getattr(config, name, default)
                for name, default in vars(default_config).items()
            }
            if legacy_config:
                values["associative_memory_enabled"] = False
            self.config = SynchronousControlConfig(**values)
        self.encoder = CausalVisualEncoder(self.config)
        self.fusion = PerceptionFusion(self.config)
        self.associative_memory = FixedAssociativeMemory(self.config)
        self.memory = TokenPipelineMemory(self.config)
        self.decoder = ResponseDecoder(self.config)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if not any(name.startswith("associative_memory.") for name in state_dict):
            strict = False
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def init_state(
        self, batch: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> ControlState:
        state = self.memory.init_state(batch, device)
        state.associative = self.associative_memory.init_state(batch, device, dtype)
        return state

    @staticmethod
    def save_associative_memory(path: str, state: ControlState) -> None:
        if state.associative is None:
            raise ValueError("ControlState has no associative memory")
        torch.save(state.associative.detach(), path)

    @staticmethod
    def load_associative_memory(
        path: str, device: torch.device
    ) -> AssociativeMemoryState:
        memory = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(memory, AssociativeMemoryState):
            raise TypeError("File does not contain AssociativeMemoryState")
        return memory.to(device)

    def associative_memory_statistics(self, state: ControlState) -> dict[str, float | int]:
        if state.associative is None:
            return {"capacity": self.config.associative_memory_capacity, "occupied": 0}
        memory = state.associative
        occupied = memory.occupied
        return {
            "capacity": int(memory.keys.shape[1]),
            "occupied": int(occupied.sum().item()),
            "candidate": int((occupied & (memory.tier == 0)).sum().item()),
            "stable": int((occupied & (memory.tier == 1)).sum().item()),
            "protected": int((occupied & (memory.tier == 2)).sum().item()),
            "mean_confidence": (
                float(memory.confidence[occupied].mean().item()) if occupied.any() else 0.0
            ),
        }

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
        update_associative_memory: bool = True,
        memory_salience: Tensor | None = None,
    ) -> tuple[Tensor, ControlState]:
        if video.ndim != 5 or video.shape[1] == 0:
            raise ValueError("video must be non-empty [B,T,C,H,W]")
        if state is None:
            state = self.init_state(video.shape[0], video.device, video.dtype)
        elif state.associative is None:
            # Backward compatibility for states created before associative
            # memory was introduced.
            state.associative = self.associative_memory.init_state(
                video.shape[0], video.device, video.dtype
            )
        if state.associative.keys.shape[0] != video.shape[0]:
            raise ValueError(
                "Associative memory batch size differs from video batch size"
            )
        if state.associative.keys.shape[1:] != (
            self.config.associative_memory_capacity, self.config.token_dim
        ):
            raise ValueError("Associative memory shape differs from model config")
        video, auxiliary = self._join_pending(video, auxiliary, state)
        group = self.config.frames_per_step
        complete = (video.shape[1] // group) * group
        pending_video = video[:, complete:]
        pending_auxiliary = None if auxiliary is None else auxiliary[:, complete:]

        if complete == 0:
            empty = video.new_empty((video.shape[0], 0, self.config.response_dim))
            return empty, ControlState(
                state.layers, state.step, pending_video, pending_auxiliary, state.associative
            )

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
        associative = state.associative
        for timestep in range(perceptions.shape[1]):
            perception = perceptions[:, timestep]
            if self.config.associative_memory_enabled:
                recalled, query, proposed_value, similarities = self.associative_memory.retrieve(
                    perception, associative
                )
                current_summary = self.associative_memory.current_tokens(proposed_value)
                attended_perception = torch.cat(
                    [perception, current_summary, recalled], dim=1
                )
            else:
                attended_perception = perception
            readout, running = self.memory.forward_step(attended_perception, running)
            outputs.append(self.decoder(readout))
            if self.config.associative_memory_enabled and update_associative_memory:
                salience = None
                if memory_salience is not None:
                    if memory_salience.shape[:2] != perceptions.shape[:2]:
                        raise ValueError(
                            "memory_salience must be [B, internal_time]"
                        )
                    salience = memory_salience[:, timestep]
                associative = self.associative_memory.write(
                    associative, query.detach(), proposed_value.detach(),
                    similarities.detach(), salience,
                )
        next_state = ControlState(
            running.layers, running.step, pending_video, pending_auxiliary, associative
        )
        return torch.stack(outputs, dim=1), next_state

    def forward(self, video: Tensor, auxiliary: Tensor | None = None) -> Tensor:
        responses, state = self.forward_chunk(video, auxiliary)
        if state.pending_video is not None and state.pending_video.shape[1]:
            raise ValueError("Full forward input length must be divisible by frames_per_step")
        return responses
