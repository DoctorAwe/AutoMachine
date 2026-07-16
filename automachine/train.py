"""Minimal synchronous perception-to-response training loop."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import SynchronousControlConfig, SynchronousControlProcessor
from .synthetic import MovingTargetControlDataset


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the synchronous control smoke model")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--time", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/control_smoke.pt"))
    return parser


def train(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    config = SynchronousControlConfig(
        image_size=args.image_size,
        response_dim=4,
        frames_per_step=1,
        token_dim=args.token_dim,
        state_tokens=(8, 12, 16),
        response_activation="tanh",
    )
    dataset = MovingTargetControlDataset(
        samples=max(args.steps * args.batch_size, 256),
        time=args.time,
        image_size=args.image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = SynchronousControlProcessor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()

    step = 0
    model.train()
    while step < args.steps:
        for video, target in loader:
            video, target = video.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(video)
            amplitude_loss = loss_fn(prediction, target)
            derivative_loss = loss_fn(torch.diff(prediction, dim=1), torch.diff(target, dim=1))
            loss = amplitude_loss + 0.1 * derivative_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            print(f"step={step:04d} loss={loss.item():.6f} device={device}")
            if step >= args.steps:
                break

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, "steps": step}, args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
