"""Train the synchronous processor on prepared ActionSense S04 shards."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import SynchronousControlConfig, SynchronousControlProcessor
from .train import choose_device


def shard_batches(path: Path, batch_size: int, shuffle: bool):
    shards = sorted(path.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No prepared shards found in {path}")
    if shuffle:
        random.shuffle(shards)
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            frames = np.asarray(data["frames"])
            responses = np.asarray(data["responses"], dtype=np.float32)
        indexes = np.arange(len(frames))
        if shuffle:
            np.random.shuffle(indexes)
        for start in range(0, len(indexes) - batch_size + 1, batch_size):
            selected = indexes[start : start + batch_size]
            video = torch.from_numpy(frames[selected]).permute(0, 1, 4, 2, 3).float().div_(255.0)
            target = torch.from_numpy(responses[selected])
            yield video, target


@torch.inference_mode()
def evaluate(
    model: SynchronousControlProcessor,
    validation_dir: Path,
    batch_size: int,
    device: torch.device,
    max_batches: int,
) -> tuple[float, float]:
    model.eval()
    model_error = 0.0
    mean_baseline_error = 0.0
    batches = 0
    for video, target in shard_batches(validation_dir, batch_size, shuffle=False):
        video, target = video.to(device), target.to(device)
        prediction = model(video)
        model_error += torch.mean((prediction - target) ** 2).item()
        # Responses are normalized with train-only statistics, so zero is the
        # per-channel training-mean baseline.
        mean_baseline_error += torch.mean(target**2).item()
        batches += 1
        if batches >= max_batches:
            break
    model.train()
    return model_error / batches, mean_baseline_error / batches


def save_checkpoint(
    path: Path,
    model: SynchronousControlProcessor,
    optimizer: torch.optim.Optimizer,
    config: SynchronousControlConfig,
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
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest["response_dim"]) != 16:
        raise ValueError("Expected bilateral 16-channel EMG responses")
    device = choose_device(args.device)
    config = SynchronousControlConfig(
        image_size=int(manifest["image_size"]),
        response_dim=16,
        frames_per_step=1,
        token_dim=args.token_dim,
        spatial_grid=args.spatial_grid,
        state_tokens=tuple(args.state_tokens),
        num_heads=args.num_heads,
        mlp_ratio=2,
        dropout=args.dropout,
    )
    model = SynchronousControlProcessor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amplitude_loss = nn.SmoothL1Loss()
    step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint.get("steps", 0))
        print(f"resumed {args.resume} at step={step}")

    print(f"device={device} config={config}")
    print(f"windows={manifest['windows']}")
    model.train()
    running_loss = 0.0
    running_count = 0
    try:
        while step < args.steps:
            for video, target in shard_batches(args.data / "train", args.batch_size, shuffle=True):
                video, target = video.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(video)
                loss_signal = amplitude_loss(prediction, target)
                loss_delta = amplitude_loss(torch.diff(prediction, dim=1), torch.diff(target, dim=1))
                loss = loss_signal + args.derivative_weight * loss_delta
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                step += 1
                running_loss += loss.item()
                running_count += 1

                if step % args.log_every == 0:
                    print(f"step={step:05d} train_loss={running_loss/running_count:.6f}")
                    running_loss = 0.0
                    running_count = 0
                if step % args.eval_every == 0:
                    val_mse, baseline_mse = evaluate(
                        model, args.data / "validation", args.batch_size, device, args.validation_batches
                    )
                    improvement = 100.0 * (baseline_mse - val_mse) / max(baseline_mse, 1e-12)
                    print(
                        f"step={step:05d} val_mse={val_mse:.6f} "
                        f"mean_baseline={baseline_mse:.6f} improvement={improvement:+.2f}%"
                    )
                if args.save_every and step % args.save_every == 0:
                    save_checkpoint(args.checkpoint, model, optimizer, config, step, args)
                    print(f"saved checkpoint to {args.checkpoint}")
                if step >= args.steps:
                    break
    except KeyboardInterrupt:
        save_checkpoint(args.checkpoint, model, optimizer, config, step, args)
        print(f"interrupted; saved checkpoint at step={step} to {args.checkpoint}")
        return

    save_checkpoint(args.checkpoint, model, optimizer, config, step, args)
    print(f"finished; saved checkpoint to {args.checkpoint}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train video-to-EMG on ActionSense S04")
    parser.add_argument("--data", type=Path, default=Path("data/actionsense/S04/processed"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--spatial-grid", type=int, default=4)
    parser.add_argument("--state-tokens", type=int, nargs="+", default=[16, 16, 24, 32])
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--derivative-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/actionsense_S04.pt"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--validation-batches", type=int, default=8)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
