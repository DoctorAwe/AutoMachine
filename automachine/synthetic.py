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


class EpisodicAssociationDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Episode-specific cue associations that cannot be solved by fixed weights.

    Four visual cues are randomly permuted onto four auxiliary/response classes
    in every episode. After a gap longer than the token pipeline, one cue is
    shown without its auxiliary label and must recall that episode's pairing.
    """

    def __init__(
        self,
        samples: int = 4096,
        image_size: int = 24,
        cue_frames: int = 2,
        gap: int = 48,
        query_frames: int = 2,
        seed_offset: int = 0,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.cue_frames = cue_frames
        self.gap = gap
        self.query_frames = query_frames
        self.seed_offset = seed_offset
        axis = torch.linspace(-1.0, 1.0, image_size)
        self.yy, self.xx = torch.meshgrid(axis, axis, indexing="ij")

    def __len__(self) -> int:
        return self.samples

    def _cue(self, cue: int, generator: torch.Generator) -> Tensor:
        jitter = (torch.rand(2, generator=generator) - 0.5) * 0.25
        x, y = self.xx - jitter[0], self.yy - jitter[1]
        if cue == 0:
            shape = ((x.square() + y.square()) < 0.22).float()
        elif cue == 1:
            shape = ((x.abs() < 0.48) & (y.abs() < 0.48)).float()
        elif cue == 2:
            shape = ((y > -0.55) & (y < 0.65 - 1.4 * x.abs())).float()
        else:
            shape = ((x.abs() + y.abs()) < 0.62).float()
        colors = torch.tensor(
            (
                (1.0, 0.15, 0.10),
                (0.10, 0.35, 1.0),
                (0.15, 0.90, 0.25),
                (0.90, 0.75, 0.10),
            )
        )
        background = 0.03 + torch.rand((), generator=generator) * 0.05
        frame = background + colors[cue, :, None, None] * shape
        noise = torch.randn(frame.shape, generator=generator) * 0.015
        return (frame + noise).clamp(0.0, 1.0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        generator = torch.Generator().manual_seed(self.seed_offset + index)
        association = torch.randperm(4, generator=generator)
        query_cue = int(torch.randint(0, 4, (), generator=generator))
        frames: list[Tensor] = []
        auxiliary: list[Tensor] = []
        salience: list[Tensor] = []

        for cue in range(4):
            cue_frame = self._cue(cue, generator)
            label = torch.nn.functional.one_hot(association[cue], 4).float()
            for _ in range(self.cue_frames):
                frames.append(cue_frame)
                auxiliary.append(label)
                salience.append(torch.tensor(1.0))

        for _ in range(self.gap):
            frames.append(torch.rand(3, self.image_size, self.image_size, generator=generator) * 0.08)
            auxiliary.append(torch.zeros(4))
            salience.append(torch.tensor(0.0))

        query_frame = self._cue(query_cue, generator)
        for _ in range(self.query_frames):
            frames.append(query_frame)
            auxiliary.append(torch.zeros(4))
            salience.append(torch.tensor(0.0))
        return (
            torch.stack(frames),
            torch.stack(auxiliary),
            association[query_cue].long(),
            torch.stack(salience),
        )
