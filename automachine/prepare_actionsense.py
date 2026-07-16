"""Prepare one ActionSense subject as aligned video-to-EMG training shards."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np


@dataclass(frozen=True)
class Activity:
    name: str
    start: float
    stop: float


def _decode(value) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def load_activities(h5: h5py.File) -> list[Activity]:
    group = h5["experiment-activities"]["activities"]
    rows = np.asarray(group["data"])
    times = np.asarray(group["time_s"]).squeeze()
    open_events: dict[str, tuple[float, str]] = {}
    activities: list[Activity] = []
    for row, timestamp in zip(rows, times):
        name, marker, rating, _notes = (_decode(item) for item in row[:4])
        if rating in {"Bad", "Maybe"}:
            continue
        if marker == "Start":
            open_events[name] = (float(timestamp), rating)
        elif marker == "Stop" and name in open_events:
            start, _ = open_events.pop(name)
            if float(timestamp) > start:
                activities.append(Activity(name, start, float(timestamp)))
    return sorted(activities, key=lambda item: item.start)


def find_video_times(h5: h5py.File) -> np.ndarray:
    candidates: list[tuple[int, str, np.ndarray]] = []

    def visit(name: str, obj) -> None:
        lower = name.lower()
        if not isinstance(obj, h5py.Dataset) or "video-world" not in lower:
            return
        if not (lower.endswith("time_s") or "frame_timestamp" in lower):
            return
        values = np.asarray(obj).squeeze()
        if values.ndim != 1 or len(values) < 2 or not np.issubdtype(values.dtype, np.number):
            return
        # Epoch seconds are preferred over device-local timestamps.
        epoch_score = 2 if float(np.nanmedian(values)) > 1_000_000_000 else 0
        time_score = 1 if lower.endswith("time_s") else 0
        candidates.append((epoch_score + time_score, name, values.astype(np.float64)))

    h5.visititems(visit)
    if not candidates:
        raise KeyError("Could not find first-person world-video timestamps in HDF5")
    _, name, times = max(candidates, key=lambda item: (item[0], len(item[2])))
    if not np.all(np.diff(times) >= 0):
        raise ValueError(f"Video timestamps are not monotonic: {name}")
    print(f"video timestamps: {name} shape={times.shape}")
    return times


def load_emg(h5: h5py.File, device: str) -> tuple[np.ndarray, np.ndarray]:
    group = h5[device]["emg"]
    signal = np.asarray(group["data"], dtype=np.float32)
    times = np.asarray(group["time_s"], dtype=np.float64).squeeze()
    if signal.ndim != 2 or signal.shape[1] != 8 or len(signal) != len(times):
        raise ValueError(f"Unexpected {device} EMG shape: {signal.shape}, times={times.shape}")
    return signal, times


def rms_at_times(signal: np.ndarray, signal_times: np.ndarray, target_times: np.ndarray, half_width: float) -> np.ndarray:
    squared = signal.astype(np.float64) ** 2
    prefix = np.vstack([np.zeros((1, signal.shape[1])), np.cumsum(squared, axis=0)])
    left = np.searchsorted(signal_times, target_times - half_width, side="left")
    right = np.searchsorted(signal_times, target_times + half_width, side="right")
    counts = np.maximum(right - left, 1)[:, None]
    return np.sqrt((prefix[right] - prefix[left]) / counts).astype(np.float32)


def split_activities(activities: list[Activity]) -> dict[str, list[Activity]]:
    count = len(activities)
    if count < 5:
        raise ValueError(f"Need at least 5 valid activity instances, found {count}")
    train_end = max(1, int(count * 0.70))
    validation_end = max(train_end + 1, int(count * 0.85))
    validation_end = min(validation_end, count - 1)
    return {
        "train": activities[:train_end],
        "validation": activities[train_end:validation_end],
        "test": activities[validation_end:],
    }


class ShardWriter:
    def __init__(self, root: Path, shard_size: int, metadata: dict) -> None:
        self.root = root
        self.shard_size = shard_size
        self.metadata = metadata
        self.frames: list[np.ndarray] = []
        self.responses: list[np.ndarray] = []
        self.labels: list[str] = []
        self.index = 0
        root.mkdir(parents=True, exist_ok=True)

    def add(self, frames: np.ndarray, responses: np.ndarray, label: str) -> None:
        self.frames.append(frames)
        self.responses.append(responses)
        self.labels.append(label)
        if len(self.frames) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.frames:
            return
        path = self.root / f"shard_{self.index:04d}.npz"
        np.savez_compressed(
            path,
            frames=np.stack(self.frames),
            responses=np.stack(self.responses),
            labels=np.asarray(self.labels),
            **self.metadata,
        )
        print(f"wrote {path} windows={len(self.frames)}")
        self.frames.clear()
        self.responses.clear()
        self.labels.clear()
        self.index += 1


def prepare(args: argparse.Namespace) -> None:
    if args.output.exists():
        shutil.rmtree(args.output)
    temporary = args.output / "_episodes"
    temporary.mkdir(parents=True)

    with h5py.File(args.hdf5, "r") as h5:
        activities = load_activities(h5)
        video_times = find_video_times(h5)
        left, left_times = load_emg(h5, "myo-left")
        right, right_times = load_emg(h5, "myo-right")

    common_start = max(video_times[0], left_times[0], right_times[0])
    common_stop = min(video_times[-1], left_times[-1], right_times[-1])
    activities = [item for item in activities if item.start >= common_start and item.stop <= common_stop]
    splits = split_activities(activities)
    activity_to_split = {id(item): split for split, items in splits.items() for item in items}

    rate = 1.0 / float(np.median(np.diff(video_times)))
    half_width = 0.5 / rate
    left_rms = rms_at_times(left, left_times, video_times, half_width)
    right_rms = rms_at_times(right, right_times, video_times, half_width)
    emg = np.concatenate([left_rms, right_rms], axis=1)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {args.video}")
    episode_records: list[tuple[str, str, Path]] = []
    activity_index = 0
    activity_frames: list[np.ndarray] = []
    activity_responses: list[np.ndarray] = []

    def flush_activity(index: int) -> None:
        if index >= len(activities) or len(activity_frames) < args.window:
            activity_frames.clear()
            activity_responses.clear()
            return
        activity = activities[index]
        split = activity_to_split[id(activity)]
        path = temporary / f"episode_{index:03d}.npz"
        np.savez(
            path,
            frames=np.asarray(activity_frames, dtype=np.uint8),
            responses=np.asarray(activity_responses, dtype=np.float32),
        )
        episode_records.append((split, activity.name, path))
        activity_frames.clear()
        activity_responses.clear()

    frame_index = 0
    while frame_index < len(video_times):
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = video_times[frame_index]
        while activity_index < len(activities) and timestamp > activities[activity_index].stop:
            flush_activity(activity_index)
            activity_index += 1
        if activity_index < len(activities):
            activity = activities[activity_index]
            if activity.start <= timestamp <= activity.stop:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(
                    frame, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA
                )
                activity_frames.append(frame)
                activity_responses.append(emg[frame_index])
        frame_index += 1
    capture.release()
    if activity_index < len(activities):
        flush_activity(activity_index)
    print(
        f"decoded frames={frame_index} estimated_rate={rate:.2f}Hz "
        f"valid_episodes={len(episode_records)}"
    )

    train_values: list[np.ndarray] = []
    for split, _, path in episode_records:
        if split == "train":
            with np.load(path) as episode:
                train_values.append(np.asarray(episode["responses"], dtype=np.float32))
    if not train_values:
        raise ValueError("No train episodes survived filtering")
    train_concat = np.concatenate(train_values)
    mean = train_concat.mean(axis=0).astype(np.float32)
    std = train_concat.std(axis=0).clip(min=1e-6).astype(np.float32)
    metadata = {"response_mean": mean, "response_std": std}
    writers = {
        split: ShardWriter(args.output / split, args.shard_size, metadata)
        for split in ("train", "validation", "test")
    }
    counts = {split: 0 for split in writers}
    for split, label, path in episode_records:
        with np.load(path) as episode:
            episode_frames = episode["frames"]
            episode_responses = (episode["responses"] - mean) / std
            for start in range(0, len(episode_frames) - args.window + 1, args.stride):
                stop = start + args.window
                writers[split].add(episode_frames[start:stop], episode_responses[start:stop], label)
                counts[split] += 1
    for writer in writers.values():
        writer.flush()
    shutil.rmtree(temporary)

    manifest = {
        "subject": "S04",
        "image_size": args.image_size,
        "window": args.window,
        "stride": args.stride,
        "response_dim": 16,
        "video_rate_hz": rate,
        "response_mean": mean.tolist(),
        "response_std": std.tolist(),
        "windows": counts,
        "activities": {split: [item.name for item in items] for split, items in splits.items()},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ActionSense S04 video-to-EMG shards")
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/actionsense/S04/processed"))
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=32)
    return parser


def main() -> None:
    prepare(build_parser().parse_args())


if __name__ == "__main__":
    main()
