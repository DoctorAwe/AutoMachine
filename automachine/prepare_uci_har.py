"""Convert the extracted UCI HAR raw inertial signals to AutoMachine NPZ files.

The source archive is intentionally not downloaded by this module. Download and
extract it from the official UCI page, then run::

    python -m automachine.prepare_uci_har --source "UCI HAR Dataset" --output data/uci_har
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SIGNALS = (
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
)


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal_dir = root / split / "Inertial Signals"
    channels = [
        np.loadtxt(signal_dir / f"{name}_{split}.txt", dtype=np.float32)
        for name in SIGNALS
    ]
    sequences = np.stack(channels, axis=-1)  # [window, 128 samples, 9 channels]
    labels = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(root / split / f"subject_{split}.txt", dtype=np.int64)
    return sequences, labels, subjects


def convert(source: Path, output: Path) -> None:
    train, train_labels, train_subjects = load_split(source, "train")
    test, test_labels, test_subjects = load_split(source, "test")

    # Fit normalization on train only to avoid test-set leakage.
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True).clip(min=1e-6)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(feature_names=np.asarray(SIGNALS), mean=mean.squeeze(), std=std.squeeze())
    np.savez_compressed(
        output / "train.npz",
        sequences=(train - mean) / std,
        labels=train_labels,
        subjects=train_subjects,
        **metadata,
    )
    np.savez_compressed(
        output / "test.npz",
        sequences=(test - mean) / std,
        labels=test_labels,
        subjects=test_subjects,
        **metadata,
    )
    print(f"wrote {len(train)} train and {len(test)} test sequences to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert UCI HAR raw signals to NPZ")
    parser.add_argument("--source", type=Path, required=True, help="Extracted 'UCI HAR Dataset' folder")
    parser.add_argument("--output", type=Path, default=Path("data/uci_har"))
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
