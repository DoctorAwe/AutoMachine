"""Train/evaluate Li 2025 natural-movie to continuous SC calcium response."""

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
from .train_goldin2022 import neuron_correlations, finite_summary, streaming_predict


def batches(path: Path, batch_size: int, shuffle: bool = True):
    shards = sorted(path.glob("shard_*.npz"))
    if shuffle:
        random.shuffle(shards)
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            frames = np.asarray(data["frames"], dtype=np.float32) / 255.0
            responses = np.asarray(data["responses"], dtype=np.float32)
        order = np.arange(len(frames))
        if shuffle:
            np.random.shuffle(order)
        for offset in range(0, len(order), batch_size):
            take = order[offset:offset + batch_size]
            yield torch.from_numpy(frames[take]).permute(0, 1, 4, 2, 3), torch.from_numpy(responses[take])


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.reshape(-1, prediction.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    prediction = prediction - prediction.mean(0, keepdim=True)
    target = target - target.mean(0, keepdim=True)
    denominator = torch.sqrt(prediction.square().sum(0) * target.square().sum(0) + 1e-8)
    valid = target.square().sum(0) > 1e-8
    if not torch.any(valid):
        return prediction.new_zeros(())
    return 1.0 - ((prediction * target).sum(0) / denominator)[valid].mean()


def delta_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape[1] < 2:
        return prediction.new_zeros(())
    return F.smooth_l1_loss(torch.diff(prediction, dim=1), torch.diff(target, dim=1))


def load_episodes(path: Path) -> list[tuple[torch.Tensor, torch.Tensor]]:
    records: dict[int, list] = {}
    for shard in sorted(path.glob("shard_*.npz")):
        with np.load(shard, allow_pickle=False) as data:
            for episode, start, frames, responses in zip(data["episode"], data["starts"], data["frames"], data["responses"]):
                records.setdefault(int(episode), []).append((int(start), frames, responses))
    episodes = []
    for episode in sorted(records):
        rows = sorted(records[episode], key=lambda item: item[0])
        total = max(start + len(frames) for start, frames, _ in rows)
        frames = np.zeros((total, *rows[0][1].shape[1:]), dtype=np.float32)
        responses = np.zeros((total, rows[0][2].shape[-1]), dtype=np.float32)
        filled = np.zeros(total, dtype=bool)
        for start, frame, response in rows:
            select = ~filled[start:start + len(frame)]
            frames[start:start + len(frame)][select] = frame[select] / 255.0
            responses[start:start + len(frame)][select] = response[select]
            filled[start:start + len(frame)] = True
        if not np.all(filled):
            raise ValueError(f"episode {episode} has gaps; prepare with stride <= window")
        episodes.append((torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0), torch.from_numpy(responses).unsqueeze(0)))
    if not episodes:
        raise FileNotFoundError(f"No prepared episodes in {path}")
    return episodes


@torch.inference_mode()
def evaluate(model, path: Path, device: torch.device, chunks: tuple[int, ...], max_steps: int = 0) -> dict:
    model.eval()
    predictions, targets = [], []
    causal_errors, stream_errors = [], []
    for video, target in load_episodes(path):
        if max_steps:
            video, target = video[:, :max_steps], target[:, :max_steps]
        video, target = video.to(device), target.to(device)
        prediction = streaming_predict(model, video, chunks)
        predictions.append(prediction.cpu()); targets.append(target.cpu())
        diagnostic = video[:, :min(32, video.shape[1])]
        full = model(diagnostic)
        streamed = streaming_predict(model, diagnostic, chunks)
        stream_errors.append((full - streamed).abs().max().item())
        boundary = max(1, diagnostic.shape[1] // 2)
        changed = diagnostic.clone(); changed[:, boundary:] = torch.flip(changed[:, boundary:], (1,))
        causal_errors.append((full[:, :boundary] - model(changed)[:, :boundary]).abs().max().item())
    prediction = torch.cat([item.squeeze(0) for item in predictions], dim=0).unsqueeze(0)
    target = torch.cat([item.squeeze(0) for item in targets], dim=0).unsqueeze(0)
    correlations = neuron_correlations(prediction, target)
    delta_correlations = neuron_correlations(torch.diff(prediction, dim=1), torch.diff(target, dim=1))
    summary, delta_summary = finite_summary(correlations), finite_summary(delta_correlations)
    per_neuron_mse = (prediction - target).square().mean((0, 1))
    baseline_mse = target.square().mean((0, 1))  # zero is the normalized training mean
    model.train()
    return {
        "mse": per_neuron_mse.mean().item(), "baseline_mse": baseline_mse.mean().item(),
        "response_correlation": summary["mean"], "response_correlation_median": summary["median"],
        "delta_correlation": delta_summary["mean"],
        "neurons_better_fraction": (per_neuron_mse < baseline_mse).float().mean().item(),
        "causal_max_error": max(causal_errors), "stream_max_error": max(stream_errors),
        "time_steps": int(target.shape[1]), "per_neuron_correlation": correlations.tolist(),
    }


def print_metrics(label: str, step: int | None, metrics: dict) -> None:
    prefix = label if step is None else f"step={step:05d} {label}"
    print(f"{prefix}_mse={metrics['mse']:.6f} baseline={metrics['baseline_mse']:.6f} "
          f"better={metrics['mse'] < metrics['baseline_mse']} response_r={metrics['response_correlation']:.4f} "
          f"delta_r={metrics['delta_correlation']:.4f} neurons_better={metrics['neurons_better_fraction']:.1%} "
          f"causal_max={metrics['causal_max_error']:.2e} stream_max={metrics['stream_max_error']:.2e}")


def save(path, model, optimizer, config, step, args, manifest, best_metric):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                "steps": step, "args": vars(args), "data_manifest": manifest, "best_validation_mse": best_metric}, path)


def train(args: argparse.Namespace) -> None:
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("dataset", "").startswith("Li et al. 2025"):
        raise ValueError("Data was not prepared by prepare_li2025")
    device = choose_device(args.device)
    config = SynchronousControlConfig(image_size=int(manifest["image_size"]), input_channels=int(manifest["input_channels"]),
        response_dim=int(manifest["response_dim"]), token_dim=args.token_dim, spatial_grid=args.spatial_grid,
        state_tokens=tuple(args.state_tokens), num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        associative_memory_capacity=args.memory_capacity,
        associative_value_tokens=args.memory_value_tokens,
        associative_top_k=args.memory_top_k,
        associative_retrieval_threshold=args.memory_retrieval_threshold,
        associative_write_threshold=args.memory_write_threshold)
    model = SynchronousControlProcessor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step, best = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError:
            print("checkpoint predates the associative-memory parameters; using a fresh optimizer")
        step = int(checkpoint.get("steps", 0)); best = float(checkpoint.get("best_validation_mse", best))
        for group in optimizer.param_groups: group["lr"] = args.lr; group["weight_decay"] = args.weight_decay
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,} neurons={config.response_dim} layers={len(config.state_tokens)}")
    running = [0.0, 0.0, 0.0, 0]
    try:
        while step < args.steps:
            for video, target in batches(args.data / "train", args.batch_size):
                video, target = video.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(video)
                amplitude = F.smooth_l1_loss(prediction, target)
                correlation = correlation_loss(prediction, target)
                delta = delta_loss(prediction, target)
                loss = amplitude + args.correlation_weight * correlation + args.delta_weight * delta
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); optimizer.step()
                step += 1
                running[0] += loss.item(); running[1] += correlation.item(); running[2] += delta.item(); running[3] += 1
                if step % args.log_every == 0:
                    print(f"step={step:05d} train_loss={running[0]/running[3]:.6f} corr_loss={running[1]/running[3]:.6f} delta_loss={running[2]/running[3]:.6f}")
                    running = [0.0, 0.0, 0.0, 0]
                if step % args.eval_every == 0:
                    metrics = evaluate(model, args.data / "validation", device, tuple(args.stream_chunks), args.evaluation_max_steps)
                    print_metrics("val", step, metrics)
                    (args.checkpoint.parent / f"{args.checkpoint.stem}_validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                    if metrics["mse"] < best:
                        best = metrics["mse"]
                        save(args.checkpoint, model, optimizer, config, step, args, manifest, best)
                        print(f"saved best checkpoint to {args.checkpoint}")
                if step >= args.steps: break
    except KeyboardInterrupt:
        save(args.checkpoint.with_name(args.checkpoint.stem + "_last.pt"), model, optimizer, config, step, args, manifest, best)
        print("interrupted; saved last checkpoint"); return
    metrics = evaluate(model, args.data / "test", device, tuple(args.stream_chunks), args.evaluation_max_steps)
    print_metrics("test", None, metrics)
    save(args.checkpoint.with_name(args.checkpoint.stem + "_last.pt"), model, optimizer, config, step, args, manifest, best)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Li 2025 natural-movie SC responses")
    parser.add_argument("--data", type=Path, default=Path("data/li2025/processed")); parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--spatial-grid", type=int, default=4); parser.add_argument("--state-tokens", type=int, nargs="+", default=list(DEFAULT_STATE_TOKENS))
    parser.add_argument("--num-heads", type=int, default=8); parser.add_argument("--mlp-ratio", type=int, default=2); parser.add_argument("--dropout", type=float, default=.05)
    parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--correlation-weight", type=float, default=.25); parser.add_argument("--delta-weight", type=float, default=.1)
    parser.add_argument("--memory-capacity", type=int, default=256)
    parser.add_argument("--memory-value-tokens", type=int, default=8)
    parser.add_argument("--memory-top-k", type=int, default=4)
    parser.add_argument("--memory-retrieval-threshold", type=float, default=.72)
    parser.add_argument("--memory-write-threshold", type=float, default=.65)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/li2025_sc_16layer.pt")); parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-every", type=int, default=20); parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--stream-chunks", type=int, nargs="+", default=[8, 13, 7]); parser.add_argument("--evaluation-max-steps", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.correlation_weight < 0 or args.delta_weight < 0: raise SystemExit("loss weights must be nonnegative")
    train(args)


if __name__ == "__main__": main()
