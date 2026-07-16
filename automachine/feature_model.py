"""Feature-stream variant of the hierarchical neural processor.

This path is for already-extracted temporal signals such as pose keypoints,
EMG channels, calcium traces, spike counts, or other vector streams:

    [batch, time, input_features] -> [batch, output_steps, output_features]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .model import HierarchicalMemoryPipeline, NeuralStreamProcessorConfig, TokenResampler


@dataclass(frozen=True)
class FeatureStreamProcessorConfig:
    input_features: int
    output_features: int
    chunk_size: int = 8
    token_dim: int = 128
    tokens_per_step: int = 16
    state_tokens: tuple[int, ...] = (32, 64, 96, 128)
    num_heads: int = 4
    mlp_ratio: int = 4

    def __post_init__(self) -> None:
        if self.input_features <= 0 or self.output_features <= 0:
            raise ValueError("input_features and output_features must be positive")
        if self.chunk_size <= 0 or self.tokens_per_step <= 0:
            raise ValueError("chunk_size and tokens_per_step must be positive")
        if self.token_dim <= 0 or self.num_heads <= 0 or self.token_dim % self.num_heads:
            raise ValueError("token_dim must be positive and divisible by num_heads")
        if not self.state_tokens or any(count <= 0 for count in self.state_tokens):
            raise ValueError("state_tokens must contain positive token counts")

    def to_memory_config(self) -> NeuralStreamProcessorConfig:
        return NeuralStreamProcessorConfig(
            image_size=1,
            input_channels=1,
            chunk_size=1,
            token_dim=self.token_dim,
            encoder_grid=1,
            state_tokens=self.state_tokens,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            output_channels=1,
        )


class FeatureStreamEncoder(nn.Module):
    """Encodes a temporal feature chunk into a compact token set."""

    def __init__(self, config: FeatureStreamProcessorConfig) -> None:
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.input_features, config.token_dim)
        self.temporal_pos = nn.Parameter(torch.zeros(1, config.chunk_size, config.token_dim))
        self.resampler = TokenResampler(config.tokens_per_step, config.token_dim, config.num_heads)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def forward(self, chunk: Tensor) -> Tensor:
        if chunk.ndim != 3:
            raise ValueError(f"Expected [B, chunk, F], got {tuple(chunk.shape)}")
        if chunk.shape[1] != self.config.chunk_size:
            raise ValueError(f"Expected chunk_size={self.config.chunk_size}, got {chunk.shape[1]}")
        x = self.input(chunk) + self.temporal_pos
        return self.resampler(x)


class FeatureStreamDecoder(nn.Module):
    """Decodes memory readout tokens into one output feature vector."""

    def __init__(self, config: FeatureStreamProcessorConfig) -> None:
        super().__init__()
        self.readout = TokenResampler(1, config.token_dim, config.num_heads)
        self.output = nn.Sequential(
            nn.LayerNorm(config.token_dim),
            nn.Linear(config.token_dim, config.token_dim),
            nn.GELU(),
            nn.Linear(config.token_dim, config.output_features),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        token = self.readout(tokens).squeeze(1)
        return self.output(token)


class FeatureStreamProcessor(nn.Module):
    """End-to-end streaming model for vector-valued temporal signals."""

    def __init__(self, config: FeatureStreamProcessorConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = FeatureStreamEncoder(config)
        self.memory = HierarchicalMemoryPipeline(config.to_memory_config())
        self.decoder = FeatureStreamDecoder(config)

    def forward_step(
        self,
        chunk: Tensor,
        states: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        encoded = self.encoder(chunk)
        readout, new_states = self.memory(encoded, states)
        return self.decoder(readout), new_states

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 3:
            raise ValueError(f"Expected [B, T, F], got {tuple(sequence.shape)}")
        chunk = self.config.chunk_size
        if sequence.shape[1] < chunk:
            raise ValueError(f"Sequence needs at least {chunk} steps")

        states: list[Tensor] | None = None
        outputs: list[Tensor] = []
        for start in range(0, sequence.shape[1] - chunk + 1):
            output, states = self.forward_step(sequence[:, start : start + chunk], states)
            outputs.append(output)
        return torch.stack(outputs, dim=1)
