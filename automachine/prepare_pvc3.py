"""Inspect or prepare one CRCNS pvc-3 visual-stimulus recording.

The natural movie is stored as raw 64x64 uint8 frames.  Each selected .spk
file is a little-endian uint64 stream of spike timestamps in microseconds.
Explicit paths are intentional: pvc-3 contains several stimulus protocols and
the archive layout/recording names must not be guessed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FRAME_SIDE = 64
NEURONS = 10


def inspect(root: Path) -> None:
    files = [path for path in root.rglob("*") if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found under {root}")
    for path in files:
        relative = path.relative_to(root)
        size = path.stat().st_size
        hints: list[str] = []
        if path.suffix.lower() == ".spk":
            hints.append(f"{size // 8} uint64 timestamps")
        if size and size % (FRAME_SIDE * FRAME_SIDE) == 0:
            hints.append(f"{size // (FRAME_SIDE * FRAME_SIDE)} raw 64x64 frames")
        suffix = f"  [{' | '.join(hints)}]" if hints else ""
        print(f"{relative}  {size} bytes{suffix}")


def read_spikes(path: Path) -> np.ndarray:
    size = path.stat().st_size
    if size % 8:
        raise ValueError(f"Spike file size is not divisible by 8: {path}")
    spikes = np.fromfile(path, dtype="<u8")
    if len(spikes) > 1 and np.any(spikes[1:] < spikes[:-1]):
        raise ValueError(f"Spike timestamps are not sorted: {path}")
    return spikes


def write_shards(
    frames: np.ndarray,
    responses: np.ndarray,
    destination: Path,
    window: int,
    stride: int,
    shard_windows: int,
) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    starts = np.arange(0, len(frames) - window + 1, stride)
    written = 0
    for shard_index, offset in enumerate(range(0, len(starts), shard_windows)):
        selected = starts[offset : offset + shard_windows]
        video = np.stack([frames[start : start + window] for start in selected])
        target = np.stack([responses[start : start + window] for start in selected])
        np.savez_compressed(
            destination / f"shard_{shard_index:04d}.npz",
            frames=video[..., None],
            responses=target.astype(np.float32),
        )
        written += len(selected)
    return written


def prepare(args: argparse.Namespace) -> None:
    if len(args.spike_files) != NEURONS:
        raise ValueError(f"Exactly {NEURONS} simultaneously recorded neuron files are required")
    raw = np.fromfile(args.movie, dtype=np.uint8)
    pixels = FRAME_SIDE * FRAME_SIDE
    if len(raw) % pixels:
        raise ValueError("Movie byte count is not divisible by 64*64")
    frames = raw.reshape(-1, FRAME_SIDE, FRAME_SIDE)
    if args.frame_count is not None:
        frames = frames[: args.frame_count]
    duration_us = len(frames) * 1_000_000.0 / args.frame_rate
    edges = (
        args.onset_us
        + args.response_delay_ms * 1000.0
        + np.arange(len(frames) + 1) * 1_000_000.0 / args.frame_rate
    )
    responses = np.stack(
        [np.histogram(read_spikes(path), bins=edges)[0] for path in args.spike_files], axis=-1
    ).astype(np.float32)

    # Guard against an onset/rate mismatch that would silently create empty labels.
    if responses.sum() == 0:
        raise ValueError(
            "No spikes fell inside the aligned movie interval; check --onset-us, "
            "--frame-rate, selected recording, and timestamp units"
        )

    train_end = int(len(frames) * 0.70)
    validation_end = int(len(frames) * 0.85)
    ranges = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, len(frames)),
    }
    windows: dict[str, int] = {}
    for split, (start, stop) in ranges.items():
        windows[split] = write_shards(
            frames[start:stop], responses[start:stop], args.output / split,
            args.window, args.stride, args.shard_windows,
        )
    train_mean = responses[:train_end].mean(axis=0)
    manifest = {
        "dataset": "CRCNS pvc-3",
        "movie": str(args.movie),
        "spike_files": [str(path) for path in args.spike_files],
        "image_size": FRAME_SIDE,
        "input_channels": 1,
        "response_dim": NEURONS,
        "frame_rate_hz": args.frame_rate,
        "onset_us": args.onset_us,
        "response_delay_ms": args.response_delay_ms,
        "duration_us": duration_us,
        "frame_count": len(frames),
        "window": args.window,
        "stride": args.stride,
        "windows": windows,
        "train_mean_spikes_per_frame": train_mean.tolist(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect/prepare CRCNS pvc-3")
    parser.add_argument("--inspect-root", type=Path)
    parser.add_argument("--movie", type=Path)
    parser.add_argument("--spike-files", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, default=Path("data/pvc3/processed"))
    parser.add_argument("--frame-rate", type=float, default=30.0)
    parser.add_argument("--onset-us", type=float, default=0.0)
    parser.add_argument("--response-delay-ms", type=float, default=50.0)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--shard-windows", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.inspect_root is not None:
        inspect(args.inspect_root)
        return
    if args.movie is None or not args.spike_files:
        raise SystemExit("Preparation requires --movie and ten --spike-files entries")
    prepare(args)


if __name__ == "__main__":
    main()
