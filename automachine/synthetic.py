"""Synthetic stream data for early model bring-up.

The project goal is biological/real-world signal streams, but the first code
checkpoint needs a deterministic source that proves recurrent state, decoding,
loss, and backpropagation all work before we spend money on real training.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.utils.data import Dataset


class MovingBlobVideoDataset(Dataset[tuple[Tensor, Tensor]]):
    """Generates tiny videos of a soft blob moving with bouncing dynamics.

    Returns:
        input_video: [time - 1, channels, height, width]
        target_video: [time - chunk_size + 1, channels, height, width]

    The target length matches ``NeuralStreamProcessor.forward`` for an input
    sequence with ``time - 1`` frames and a given chunk size.
    """

    def __init__(
        self,
        samples: int = 256,
        time: int = 9,
        image_size: int = 64,
        chunk_size: int = 3,
        channels: int = 3,
        sigma: float = 3.5,
        seed: int = 13,
    ) -> None:
        if time <= chunk_size:
            raise ValueError("time must be larger than chunk_size")
        self.samples = samples
        self.time = time
        self.image_size = image_size
        self.chunk_size = chunk_size
        self.channels = channels
        self.sigma = sigma
        self.seed = seed

        axis = torch.linspace(0, image_size - 1, image_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.registered_grid = torch.stack([xx, yy], dim=0)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        video = self._make_video(generator)
        input_video = video[:-1]
        target_video = video[self.chunk_size :]
        return input_video, target_video

    def _make_video(self, generator: torch.Generator) -> Tensor:
        size = self.image_size
        margin = size * 0.18
        pos = torch.rand(2, generator=generator) * (size - 2 * margin) + margin
        angle = torch.rand((), generator=generator) * math.tau
        speed = torch.rand((), generator=generator) * 2.2 + 1.0
        vel = torch.tensor([torch.cos(angle), torch.sin(angle)]) * speed
        color = torch.rand(self.channels, generator=generator) * 0.7 + 0.3

        frames = []
        for _ in range(self.time):
            frames.append(self._render_blob(pos, color))
            pos = pos + vel
            for dim in range(2):
                if pos[dim] < margin or pos[dim] > size - margin:
                    vel[dim] = -vel[dim]
                    pos[dim] = pos[dim].clamp(margin, size - margin)
        return torch.stack(frames, dim=0)

    def _render_blob(self, pos: Tensor, color: Tensor) -> Tensor:
        grid = self.registered_grid
        dx = grid[0] - pos[0]
        dy = grid[1] - pos[1]
        blob = torch.exp(-(dx.square() + dy.square()) / (2 * self.sigma**2))
        return color[:, None, None] * blob[None, :, :]
