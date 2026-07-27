"""Prepare one Li et al. 2025 mouse superior-colliculus imaging plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


MOVIES = (
    "1_baby_owl-converted.mp4",
    "2_running_cats-converted.mp4",
    "3_foraging-converted.mp4",
    "4_optical_flow.avi",
)
# nm_temporal is stored as owl, foraging, running cats, optical flow.  These
# are response samples (about 5 Hz), despite the upstream analysis variable
# being named nm_duration.
SOURCE_SEGMENT_LENGTHS = (311, 83, 148, 35)
VIDEO_TO_SOURCE_SEGMENT = (0, 2, 1, 3)
SPLITS = {"train": (0, 1), "validation": (2,), "test": (3,)}


def load_mat(path: Path) -> dict:
    try:
        import hdf5storage
        return hdf5storage.loadmat(path)
    except Exception as first_error:
        try:
            from scipy.io import loadmat
            return loadmat(path, squeeze_me=True, struct_as_record=False)
        except Exception as second_error:
            raise RuntimeError(f"Could not read {path}: {first_error}; {second_error}") from second_error


def value(mapping: dict, name: str) -> np.ndarray:
    if name not in mapping:
        raise KeyError(f"{name!r} is missing; available keys={sorted(k for k in mapping if not k.startswith('__'))}")
    return np.asarray(mapping[name]).squeeze()


def neuron_time(array: np.ndarray, neuron_count: int, name: str) -> np.ndarray:
    array = np.asarray(array).squeeze()
    axes = [axis for axis, size in enumerate(array.shape) if size == neuron_count]
    if not axes:
        raise ValueError(f"Cannot find neuron axis in {name} shape={array.shape}, neurons={neuron_count}")
    result = np.moveaxis(array, axes[0], 0).reshape(neuron_count, -1)
    return result.astype(np.float32)


def discover(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {filename} below {root}, found {matches}")
    return matches[0]


def segment_responses(responses: np.ndarray) -> list[np.ndarray]:
    total = responses.shape[1]
    reference_total = sum(SOURCE_SEGMENT_LENGTHS)
    boundaries = np.rint(np.cumsum((0, *SOURCE_SEGMENT_LENGTHS)) * total / reference_total).astype(int)
    boundaries[-1] = total
    source = [responses[:, boundaries[i]:boundaries[i + 1]] for i in range(4)]
    return [source[index] for index in VIDEO_TO_SOURCE_SEGMENT]


def decode_at_response_rate(path: Path, steps: int, size: int, rgb: bool) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise RuntimeError(f"Could not inspect video {path}")
    wanted = np.rint(np.linspace(0, frame_count - 1, steps)).astype(int)
    frames = []
    cursor = 0
    wanted_index = 0
    while wanted_index < len(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        while wanted_index < len(wanted) and wanted[wanted_index] == cursor:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[..., None]
            frames.append(frame)
            wanted_index += 1
        cursor += 1
    capture.release()
    if len(frames) != steps:
        raise RuntimeError(f"Decoded {len(frames)}/{steps} aligned frames from {path}")
    return np.asarray(frames, dtype=np.uint8)


def window_starts(length: int, window: int, stride: int) -> np.ndarray:
    if length < window:
        return np.empty(0, dtype=np.int64)
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return np.asarray(starts, dtype=np.int64)


def write_split(output: Path, name: str, episodes: list[tuple[np.ndarray, np.ndarray]], window: int, stride: int, shard_size: int) -> int:
    destination = output / name
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    shard = 0
    count = 0
    for episode_id, (frames, responses) in enumerate(episodes):
        for start in window_starts(len(frames), window, stride):
            records.append((episode_id, int(start), frames[start:start + window], responses[start:start + window]))
            if len(records) == shard_size:
                _save_shard(destination, shard, records)
                count += len(records); shard += 1; records = []
    if records:
        _save_shard(destination, shard, records); count += len(records)
    if not count:
        raise ValueError(f"No {name} windows: reduce --window")
    return count


def _save_shard(destination: Path, index: int, records: list) -> None:
    np.savez_compressed(
        destination / f"shard_{index:04d}.npz",
        episode=np.asarray([item[0] for item in records], dtype=np.int16),
        starts=np.asarray([item[1] for item in records], dtype=np.int32),
        frames=np.asarray([item[2] for item in records], dtype=np.uint8),
        responses=np.asarray([item[3] for item in records], dtype=np.float32),
    )


def prepare(args: argparse.Namespace) -> None:
    neuron = load_mat(discover(args.root, "neuron.mat"))
    nm = load_mat(discover(args.root, "nm.mat"))
    image_ids = value(neuron, "image_id").reshape(-1)
    temporal = neuron_time(value(nm, "nm_temporal"), len(image_ids), "nm_temporal")
    unique, counts = np.unique(image_ids, return_counts=True)
    planes = {str(item): int(count) for item, count in zip(unique.tolist(), counts.tolist())}
    if args.inspect:
        print(json.dumps({"neurons": len(image_ids), "time_steps": temporal.shape[1], "image_planes": planes}, indent=2))
        return
    image_id = args.image_id if args.image_id is not None else unique[np.argmax(counts)]
    selected = np.flatnonzero(image_ids == image_id)
    if not len(selected):
        raise ValueError(f"image_id={image_id} not found; use --inspect")
    snr = value(nm, "nm_snr").reshape(-1) if "nm_snr" in nm else None
    if snr is not None and len(snr) == len(image_ids):
        selected = selected[np.isfinite(snr[selected]) & (snr[selected] >= args.min_snr)]
        if args.max_neurons and len(selected) > args.max_neurons:
            selected = selected[np.argsort(snr[selected])[-args.max_neurons:]]
    elif args.max_neurons:
        selected = selected[:args.max_neurons]
    responses = temporal[selected]
    valid = np.all(np.isfinite(responses), axis=1) & (np.std(responses, axis=1) > 1e-6)
    responses = responses[valid]
    selected = selected[valid]
    if not len(selected):
        raise ValueError("No valid neurons remain; lower --min-snr")

    response_segments = segment_responses(responses)
    episodes = []
    for filename, response in zip(MOVIES, response_segments):
        frames = decode_at_response_rate(discover(args.root, filename), response.shape[1], args.image_size, args.rgb)
        episodes.append((frames, response.T))

    train_values = np.concatenate([episodes[i][1] for i in SPLITS["train"]], axis=0)
    response_mean = train_values.mean(axis=0)
    response_std = np.maximum(train_values.std(axis=0), 1e-4)
    normalized = [(frames, (response - response_mean) / response_std) for frames, response in episodes]
    windows = {}
    for split, ids in SPLITS.items():
        windows[split] = write_split(args.output, split, [normalized[i] for i in ids], args.window, args.stride, args.shard_size)
    manifest = {
        "dataset": "Li et al. 2025 superior colliculus natural movies",
        "image_id": float(image_id), "neuron_indices": selected.tolist(),
        "response_dim": int(len(selected)), "image_size": args.image_size,
        "input_channels": 3 if args.rgb else 1, "window": args.window, "stride": args.stride,
        "movie_order": list(MOVIES), "split_movie_indices": {k: list(v) for k, v in SPLITS.items()},
        "episode_lengths": [int(item[1].shape[0]) for item in episodes],
        "response_mean": response_mean.tolist(), "response_std": response_std.tolist(), "windows": windows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"prepared image_id={image_id} neurons={len(selected)} windows={windows} at {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare one Li 2025 SC imaging plane")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/li2025/processed"))
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--image-id", type=float)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--min-snr", type=float, default=0.0)
    parser.add_argument("--max-neurons", type=int, default=128)
    parser.add_argument("--rgb", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.image_size, args.window, args.stride, args.shard_size) <= 0:
        raise SystemExit("size/window/stride/shard-size must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
