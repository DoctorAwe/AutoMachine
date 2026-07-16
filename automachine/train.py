"""Minimal training loop for the AutoMachine prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import NeuralStreamProcessor, NeuralStreamProcessorConfig
from .synthetic import MovingBlobVideoDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the neural stream processor on synthetic motion.")
    parser.add_argument("--steps", type=int, default=20, help="Optimization steps to run.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/smoke.pt"))
    parser.add_argument("--resume", type=Path, default=None, help="Resume model/optimizer state from a checkpoint.")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=0, help="Save intermediate checkpoints every N steps.")
    return parser


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return f"cuda name={name!r} capability={capability} memory={memory_gb:.1f}GB"


def save_checkpoint(
    path: Path,
    model: NeuralStreamProcessor,
    optimizer: torch.optim.Optimizer,
    config: NeuralStreamProcessorConfig,
    step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "steps": step,
            "args": vars(args),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    config = NeuralStreamProcessorConfig(
        image_size=args.image_size,
        chunk_size=args.chunk_size,
        token_dim=args.token_dim,
        state_tokens=(64, 64, 96),
        num_heads=4,
    )
    dataset = MovingBlobVideoDataset(
        samples=max(args.batch_size * args.steps, 32),
        image_size=args.image_size,
        chunk_size=args.chunk_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = NeuralStreamProcessor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn: nn.Module = nn.L1Loss()
    step = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint.get("steps", 0))
        print(f"resumed from {args.resume} at step={step}")

    print(f"device={describe_device(device)}")
    print("config=" + json.dumps(vars(args), ensure_ascii=False, default=str))

    model.train()
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
            if step % args.log_every == 0:
                print(f"step={step:04d} loss={loss.item():.6f} device={device}")
            if args.save_every > 0 and step % args.save_every == 0:
                path = args.checkpoint.with_name(f"{args.checkpoint.stem}_step_{step:06d}{args.checkpoint.suffix}")
                save_checkpoint(path, model, optimizer, config, step, args)
                print(f"saved checkpoint to {path}")
            if step >= args.steps:
                break

    save_checkpoint(args.checkpoint, model, optimizer, config, step, args)
    print(f"saved checkpoint to {args.checkpoint}")


def main() -> None:
    parser = build_parser()
    train(parser.parse_args())


if __name__ == "__main__":
    main()
