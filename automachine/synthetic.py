"""Synthetic synchronous stimulus-response data for engineering smoke tests."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.utils.data import Dataset


class MovingTargetControlDataset(Dataset[tuple[Tensor, Tensor]]):
    """Video of a moving target with its synchronous normalized control state.

    The four response channels are ``x, y, vx, vy``.  This is only a pipeline
    check; real experiments should use aligned sensor/control recordings.
    """

    def __init__(self, samples: int = 256, time: int = 16, image_size: int = 64) -> None:
        self.samples = samples
        self.time = time
        self.image_size = image_size

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        generator = torch.Generator().manual_seed(index)
        x0, y0 = torch.rand(2, generator=generator) * 1.4 - 0.7
        vx, vy = (torch.rand(2, generator=generator) * 0.12 - 0.06).tolist()
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.image_size),
            torch.linspace(-1.0, 1.0, self.image_size),
            indexing="ij",
        )
        frames: list[Tensor] = []
        responses: list[Tensor] = []
        for step in range(self.time):
            x = float(x0) + vx * step
            y = float(y0) + vy * step
            # Bounce at the visual boundary to keep the target observable.
            x = math.sin(x * math.pi / 2)
            y = math.sin(y * math.pi / 2)
            blob = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 0.015)
            frame = torch.stack((blob, blob * 0.5, 1.0 - blob * 0.25)).float()
            frames.append(frame)
            responses.append(torch.tensor((x, y, vx, vy), dtype=torch.float32))
        return torch.stack(frames), torch.stack(responses)
