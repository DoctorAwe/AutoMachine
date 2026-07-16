"""Train the feature-stream processor on NPZ feature sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .feature_model import FeatureStreamProcessor, FeatureStreamProcessorConfig
from .sequence_data import SlidingWindowSequenceDataset, load_npz_sequences
from .train import choose_device, describe_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AutoMachine on vector feature streams.")
    parser.add_argument("--data", type=Path, required=True, help="NPZ file containing a `sequences` array.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-length", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--tokens-per-step", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/features.pt"))
    return parser


def train(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    sequences = load_npz_sequences(args.data)
    features = sequences[0].shape[-1]
    dataset = SlidingWindowSequenceDataset(
        sequences=sequences,
        input_length=args.input_length,
        chunk_size=args.chunk_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    config = FeatureStreamProcessorConfig(
        input_features=features,
        output_features=features,
        chunk_size=args.chunk_size,
        token_dim=args.token_dim,
        tokens_per_step=args.tokens_per_step,
    )
    model = FeatureStreamProcessor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn: nn.Module = nn.MSELoss()

    print(f"device={describe_device(device)}")
    print("config=" + json.dumps(vars(args), ensure_ascii=False, default=str))
    print(f"loaded_sequences={len(sequences)} features={features} windows={len(dataset)}")

    step = 0
    while step < args.steps:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            step += 1
            print(f"step={step:04d} loss={loss.item():.6f} device={device}")
            if step >= args.steps:
                break

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "steps": step,
            "args": vars(args),
        },
        args.checkpoint,
    )
    print(f"saved checkpoint to {args.checkpoint}")


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
