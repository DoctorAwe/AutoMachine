"""Train the 16-stage synchronous visual-to-spike model on CRCNS pvc-3."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .model import DEFAULT_STATE_TOKENS, SynchronousControlConfig, SynchronousControlProcessor
from .train import choose_device


def batches(path: Path, batch_size: int, shuffle: bool):
    shards = sorted(path.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shards in {path}")
    if shuffle:
        random.shuffle(shards)
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            frames = np.asarray(data["frames"])
            responses = np.asarray(data["responses"], dtype=np.float32)
        indexes = np.arange(len(frames))
        if shuffle:
            np.random.shuffle(indexes)
        for offset in range(0, len(indexes) - batch_size + 1, batch_size):
            selected = indexes[offset : offset + batch_size]
            video = torch.from_numpy(frames[selected]).permute(0, 1, 4, 2, 3).float().div_(255.0)
            yield video, torch.from_numpy(responses[selected])


def poisson_loss(log_rate: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return F.poisson_nll_loss(log_rate, counts, log_input=True, full=False)


@torch.inference_mode()
def evaluate(model, path, batch_size, device, max_batches, mean_rate):
    model.eval()
    model_loss = baseline_loss = 0.0
    count = 0
    baseline = torch.log(mean_rate.clamp_min(1e-6)).view(1, 1, -1)
    for video, target in batches(path, batch_size, False):
        video, target = video.to(device), target.to(device)
        model_loss += poisson_loss(model(video), target).item()
        baseline_loss += poisson_loss(baseline.expand_as(target), target).item()
        count += 1
        if count >= max_batches:
            break
    model.train()
    return model_loss / count, baseline_loss / count


def save(path, model, optimizer, config, step, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config, "steps": step, "args": vars(args)}, path)


def train(args: argparse.Namespace) -> None:
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest["response_dim"]) != 10 or int(manifest["input_channels"]) != 1:
        raise ValueError("Expected pvc-3 grayscale input and ten neural responses")
    device = choose_device(args.device)
    config = SynchronousControlConfig(
        image_size=64, input_channels=1, response_dim=10, frames_per_step=1,
        token_dim=args.token_dim, spatial_grid=args.spatial_grid,
        state_tokens=tuple(args.state_tokens), num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio, dropout=args.dropout,
    )
    model = SynchronousControlProcessor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint.get("steps", 0))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    mean_rate = torch.tensor(manifest["train_mean_spikes_per_frame"], device=device)
    print(f"device={device} parameters={parameters:,} pipeline_layers={len(config.state_tokens)}")
    model.train()
    try:
        while step < args.steps:
            for video, target in batches(args.data / "train", args.batch_size, True):
                video, target = video.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                log_rate = model(video).clamp(max=10.0)
                loss = poisson_loss(log_rate, target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                step += 1
                if step % args.log_every == 0:
                    print(f"step={step:05d} train_poisson={loss.item():.6f}")
                if step % args.eval_every == 0:
                    val, baseline = evaluate(model, args.data / "validation", args.batch_size,
                                             device, args.validation_batches, mean_rate)
                    print(f"step={step:05d} val_poisson={val:.6f} mean_baseline={baseline:.6f} "
                          f"better={val < baseline}")
                if args.save_every and step % args.save_every == 0:
                    save(args.checkpoint, model, optimizer, config, step, args)
                if step >= args.steps:
                    break
    except KeyboardInterrupt:
        save(args.checkpoint, model, optimizer, config, step, args)
        print(f"interrupted; saved step={step} to {args.checkpoint}")
        return
    save(args.checkpoint, model, optimizer, config, step, args)
    print(f"finished; saved {args.checkpoint}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CRCNS pvc-3 visual-to-spike model")
    parser.add_argument("--data", type=Path, default=Path("data/pvc3/processed"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--spatial-grid", type=int, default=4)
    parser.add_argument("--state-tokens", type=int, nargs="+", default=list(DEFAULT_STATE_TOKENS))
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/pvc3_16layer.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--validation-batches", type=int, default=8)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
