"""Datasets for vector-valued temporal streams."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SlidingWindowSequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    """Creates next-step prediction windows from one or more feature streams.

    Args:
        sequences: List of arrays shaped [time, features].
        input_length: Number of timesteps fed to the model.
        chunk_size: Model chunk size. Determines target alignment.
        stride: Window stride.
    """

    def __init__(
        self,
        sequences: list[np.ndarray],
        input_length: int,
        chunk_size: int,
        stride: int = 1,
    ) -> None:
        if input_length < chunk_size:
            raise ValueError("input_length must be >= chunk_size")
        self.sequences = [np.asarray(seq, dtype=np.float32) for seq in sequences]
        self.input_length = input_length
        self.chunk_size = chunk_size
        self.stride = stride
        self.index: list[tuple[int, int]] = []
        for seq_id, sequence in enumerate(self.sequences):
            max_start = len(sequence) - input_length - 1
            for start in range(0, max_start + 1, stride):
                self.index.append((seq_id, start))
        if not self.index:
            raise ValueError("No windows available; check sequence length and input_length")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        seq_id, start = self.index[index]
        sequence = self.sequences[seq_id]
        inputs = sequence[start : start + self.input_length]
        targets = sequence[start + self.chunk_size : start + self.input_length + 1]
        return torch.from_numpy(inputs), torch.from_numpy(targets)


def load_npz_sequences(path: str | Path, key: str = "sequences") -> list[np.ndarray]:
    """Loads sequences from an NPZ file.

    Supported shapes:
        [samples, time, features]
        [time, features]
    """

    data = np.load(path)
    array = np.asarray(data[key], dtype=np.float32)
    if array.ndim == 2:
        return [array]
    if array.ndim == 3:
        return [array[i] for i in range(array.shape[0])]
    raise ValueError(f"Expected 2D or 3D array, got {array.shape}")
