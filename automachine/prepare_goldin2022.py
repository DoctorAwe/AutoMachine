"""Prepare one Goldin et al. 2022 retinal recording for streaming training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


STIMULUS_KEYS = {"train": "/train/stimulus", "test": "/test/stimulus"}
RESPONSE_KEYS = {"train": "/train/response/binned", "test": "/test/response/binned"}


def inspect(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        print(f"file={path}")

        def show(name: str, item: h5py.Dataset | h5py.Group) -> None:
            if isinstance(item, h5py.Dataset):
                print(f"/{name}: shape={item.shape} dtype={item.dtype}")

        handle.visititems(show)


def canonicalize(stimulus: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return stimulus [T,H,W,C] and response [T,N] without guessing by convention."""
    stimulus = np.asarray(stimulus).squeeze()
    response = np.asarray(response).squeeze()
    if stimulus.ndim not in (3, 4):
        raise ValueError(f"Expected 3D/4D stimulus, got {stimulus.shape}")
    if response.ndim != 2:
        raise ValueError(f"Expected 2D binned response, got {response.shape}")

    matches = [
        (stimulus.shape[stim_axis], stim_axis, response_axis)
        for stim_axis in range(stimulus.ndim)
        for response_axis in range(2)
        if stimulus.shape[stim_axis] == response.shape[response_axis]
    ]
    if not matches:
        raise ValueError(
            f"No common time dimension between stimulus {stimulus.shape} "
            f"and response {response.shape}"
        )
    # Time is normally much longer than either spatial side or the cell count.
    _, stimulus_time_axis, response_time_axis = max(matches, key=lambda item: item[0])
    stimulus = np.moveaxis(stimulus, stimulus_time_axis, 0)
    response = np.moveaxis(response, response_time_axis, 0)

    if stimulus.ndim == 3:
        stimulus = stimulus[..., None]
    else:
        channel_candidates = [axis for axis in range(1, 4) if stimulus.shape[axis] in (1, 3)]
        if not channel_candidates:
            raise ValueError(f"Cannot identify channel axis in stimulus {stimulus.shape}")
        stimulus = np.moveaxis(stimulus, channel_candidates[-1], -1)
    if stimulus.shape[0] != response.shape[0]:
        raise AssertionError("Canonicalized time dimensions differ")
    if stimulus.shape[-1] not in (1, 3):
        raise ValueError(f"Only grayscale/RGB stimuli are supported, got {stimulus.shape}")
    return stimulus.astype(np.float32), response.astype(np.float32)


def load_split(handle: h5py.File, split: str) -> tuple[np.ndarray, np.ndarray]:
    missing = [key for key in (STIMULUS_KEYS[split], RESPONSE_KEYS[split]) if key not in handle]
    if missing:
        raise KeyError(f"Missing required HDF5 datasets: {missing}")
    return canonicalize(handle[STIMULUS_KEYS[split]][:], handle[RESPONSE_KEYS[split]][:])


def write_shards(
    stimulus: np.ndarray,
    response: np.ndarray,
    destination: Path,
    window: int,
    stride: int,
    shard_windows: int,
) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("shard_*.npz"):
        stale.unlink()
    starts = np.arange(0, len(stimulus) - window + 1, stride)
    written = 0
    for shard_index, offset in enumerate(range(0, len(starts), shard_windows)):
        selected = starts[offset : offset + shard_windows]
        frames = np.stack([stimulus[start : start + window] for start in selected])
        targets = np.stack([response[start : start + window] for start in selected])
        np.savez_compressed(
            destination / f"shard_{shard_index:04d}.npz",
            frames=frames.astype(np.float16),
            responses=targets.astype(np.float32),
            starts=selected.astype(np.int64),
        )
        written += len(selected)
    return written


def prepare(args: argparse.Namespace) -> None:
    with h5py.File(args.session, "r") as handle:
        original_train_stimulus, original_train_response = load_split(handle, "train")
        test_stimulus, test_response = load_split(handle, "test")

    if np.any(original_train_response < 0) or np.any(test_response < 0):
        raise ValueError("Poisson targets must be non-negative")
    if not np.isfinite(original_train_stimulus).all() or not np.isfinite(original_train_response).all():
        raise ValueError("Training data contains NaN or infinity")
    if original_train_stimulus.shape[1:] != test_stimulus.shape[1:]:
        raise ValueError("Train/test stimulus shapes differ")
    if original_train_response.shape[1] != test_response.shape[1]:
        raise ValueError("Train/test neuron counts differ")

    validation_length = max(args.window, round(len(original_train_stimulus) * args.validation_fraction))
    train_length = len(original_train_stimulus) - validation_length
    if train_length < args.window:
        raise ValueError("Training sequence is too short for requested validation fraction/window")

    train_stimulus = original_train_stimulus[:train_length]
    validation_stimulus = original_train_stimulus[train_length:]
    train_response = original_train_response[:train_length]
    validation_response = original_train_response[train_length:]

    stimulus_mean = float(train_stimulus.mean(dtype=np.float64))
    stimulus_std = float(train_stimulus.std(dtype=np.float64))
    if stimulus_std < 1e-8:
        raise ValueError("Training stimulus has near-zero variance")

    def normalize(values: np.ndarray) -> np.ndarray:
        return (values - stimulus_mean) / stimulus_std

    splits = {
        "train": (normalize(train_stimulus), train_response),
        "validation": (normalize(validation_stimulus), validation_response),
        "test": (normalize(test_stimulus), test_response),
    }
    windows = {
        name: write_shards(stimulus, response, args.output / name,
                           args.window, args.stride, args.shard_windows)
        for name, (stimulus, response) in splits.items()
    }
    height, width, channels = train_stimulus.shape[1:]
    manifest = {
        "dataset": "Goldin et al. 2022",
        "session": str(args.session),
        "stimulus_keys": STIMULUS_KEYS,
        "response_keys": RESPONSE_KEYS,
        "image_height": height,
        "image_width": width,
        "input_channels": channels,
        "response_dim": int(train_response.shape[1]),
        "window": args.window,
        "stride": args.stride,
        "validation_fraction": args.validation_fraction,
        "sequence_steps": {name: len(values[0]) for name, values in splits.items()},
        "windows": windows,
        "stimulus_mean": stimulus_mean,
        "stimulus_std": stimulus_std,
        "train_mean_response": train_response.mean(axis=0).tolist(),
        "response_kind": "binned_nonnegative",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare one Goldin 2022 HDF5 session")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/goldin2022/processed"))
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--shard-windows", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise SystemExit("--validation-fraction must be between 0 and 0.5")
    if args.window <= 0 or args.stride <= 0 or args.shard_windows <= 0:
        raise SystemExit("Window, stride and shard size must be positive")
    if args.inspect:
        inspect(args.session)
    else:
        prepare(args)


if __name__ == "__main__":
    main()
