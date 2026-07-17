"""Train the 16-stage synchronous visual-to-retinal-response model."""

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
        raise FileNotFoundError(f"No prepared shards in {path}")
    if shuffle:
        random.shuffle(shards)
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            frames = np.asarray(data["frames"], dtype=np.float32)
            responses = np.asarray(data["responses"], dtype=np.float32)
        indexes = np.arange(len(frames))
        if shuffle:
            np.random.shuffle(indexes)
        for offset in range(0, len(indexes), batch_size):
            selected = indexes[offset : offset + batch_size]
            video = torch.from_numpy(frames[selected]).permute(0, 1, 4, 2, 3)
            yield video, torch.from_numpy(responses[selected])


def poisson_loss(log_rate: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    return F.poisson_nll_loss(log_rate, response, log_input=True, full=False)


def weighted_poisson_loss(
    log_rate: torch.Tensor, response: torch.Tensor, event_weight: float
) -> torch.Tensor:
    element_loss = F.poisson_nll_loss(
        log_rate, response, log_input=True, full=False, reduction="none"
    )
    weights = 1.0 + event_weight * (response > 0).to(element_loss.dtype)
    return torch.sum(element_loss * weights) / torch.sum(weights)


def temporal_correlation_loss(log_rate: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    """Differentiable mean 1-Pearson over neurons with nonconstant targets."""
    rate = torch.exp(log_rate).reshape(-1, log_rate.shape[-1])
    target = response.reshape(-1, response.shape[-1])
    rate = rate - rate.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    target_energy = torch.sum(target.square(), dim=0)
    rate_energy = torch.sum(rate.square(), dim=0)
    valid = target_energy > 1e-8
    if not torch.any(valid):
        return log_rate.new_zeros(())
    correlation = torch.sum(rate * target, dim=0) / torch.sqrt(
        rate_energy * target_energy + 1e-8
    )
    return 1.0 - correlation[valid].mean()


def temporal_delta_loss(log_rate: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    if response.shape[1] < 2:
        return log_rate.new_zeros(())
    return F.smooth_l1_loss(
        torch.diff(torch.exp(log_rate), dim=1),
        torch.diff(response, dim=1),
    )


def streaming_predict(model, video, chunk_pattern):
    """Run one sequence as a persistent stream with irregular chunk sizes."""
    outputs = []
    state = None
    cursor = 0
    pattern_index = 0
    while cursor < video.shape[1]:
        chunk_size = chunk_pattern[pattern_index % len(chunk_pattern)]
        stop = min(cursor + chunk_size, video.shape[1])
        response, state = model.forward_chunk(video[:, cursor:stop], state=state)
        outputs.append(response)
        cursor = stop
        pattern_index += 1
    return torch.cat(outputs, dim=1)


def neuron_correlations(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One Pearson correlation per neuron over one continuous interval."""
    prediction = prediction.reshape(-1, prediction.shape[-1]).float()
    target = target.reshape(-1, target.shape[-1]).float()
    prediction = prediction - prediction.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    numerator = torch.sum(prediction * target, dim=0)
    denominator = torch.sqrt(torch.sum(prediction.square(), dim=0) * torch.sum(target.square(), dim=0))
    valid = denominator > 1e-8
    correlations = torch.full_like(denominator, torch.nan)
    correlations[valid] = numerator[valid] / denominator[valid]
    return correlations


def mean_neuron_correlation(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, int]:
    """Backward-compatible summary used by unit tests and external callers."""
    correlations = neuron_correlations(prediction, target)
    valid = torch.isfinite(correlations)
    if not torch.any(valid):
        return float("nan"), 0
    return correlations[valid].mean().item(), int(valid.sum().item())


def load_continuous_split(path: Path, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct unique ordered time points from overlapping prepared windows.

    New shards contain exact start indices. Older shards remain supported by
    deriving starts from their deterministic shard/window order.
    """
    shards = sorted(path.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No prepared shards in {path}")
    records = []
    derived_window_index = 0
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            frames = np.asarray(data["frames"], dtype=np.float32)
            responses = np.asarray(data["responses"], dtype=np.float32)
            if "starts" in data:
                starts = np.asarray(data["starts"], dtype=np.int64)
            else:
                starts = np.arange(
                    derived_window_index,
                    derived_window_index + len(frames),
                    dtype=np.int64,
                ) * stride
        records.extend((int(start), frame, response) for start, frame, response in zip(starts, frames, responses))
        derived_window_index += len(frames)
    records.sort(key=lambda item: item[0])
    window = records[0][1].shape[0]
    total_steps = max(start + window for start, _, _ in records)
    frame_shape = records[0][1].shape[1:]
    response_dim = records[0][2].shape[-1]
    continuous_frames = np.empty((total_steps, *frame_shape), dtype=np.float32)
    continuous_responses = np.empty((total_steps, response_dim), dtype=np.float32)
    filled = np.zeros(total_steps, dtype=bool)
    for start, frame, response in records:
        stop = start + len(frame)
        selection = ~filled[start:stop]
        continuous_frames[start:stop][selection] = frame[selection]
        continuous_responses[start:stop][selection] = response[selection]
        filled[start:stop] = True
    if not np.all(filled):
        missing = np.flatnonzero(~filled)
        raise ValueError(
            f"Prepared windows contain gaps; first missing time index={missing[0]}. "
            "Use stride <= window and prepare the dataset again."
        )
    video = torch.from_numpy(continuous_frames).permute(0, 3, 1, 2).unsqueeze(0)
    target = torch.from_numpy(continuous_responses).unsqueeze(0)
    return video, target


def finite_summary(values: torch.Tensor) -> dict[str, float | int]:
    valid = values[torch.isfinite(values)]
    if not len(valid):
        return {"mean": float("nan"), "median": float("nan"), "p25": float("nan"), "valid": 0}
    return {
        "mean": valid.mean().item(),
        "median": valid.median().item(),
        "p25": torch.quantile(valid, 0.25).item(),
        "valid": int(len(valid)),
    }


@torch.inference_mode()
def evaluate(
    model,
    path,
    device,
    mean_response,
    chunk_pattern,
    diagnostic_batches,
    stride,
    max_steps=0,
):
    model.eval()
    video, target = load_continuous_split(path, stride)
    if max_steps:
        video, target = video[:, :max_steps], target[:, :max_steps]
    video, target = video.to(device), target.to(device)
    stream_log_rate = streaming_predict(model, video, chunk_pattern).clamp(max=10.0)
    stream_rate = torch.exp(stream_log_rate)
    baseline = torch.log(mean_response.clamp_min(1e-6)).view(1, 1, -1).expand_as(target)

    per_neuron_poisson = F.poisson_nll_loss(
        stream_log_rate, target, log_input=True, full=False, reduction="none"
    ).mean(dim=(0, 1))
    per_neuron_baseline = F.poisson_nll_loss(
        baseline, target, log_input=True, full=False, reduction="none"
    ).mean(dim=(0, 1))
    response_correlations = neuron_correlations(stream_rate, target)
    delta_correlations = neuron_correlations(
        torch.diff(stream_rate, dim=1), torch.diff(target, dim=1)
    ) if target.shape[1] > 1 else torch.full_like(response_correlations, torch.nan)
    response_summary = finite_summary(response_correlations)
    delta_summary = finite_summary(delta_correlations)
    prediction_std = stream_rate.std(dim=(0, 1), unbiased=False)
    target_std = target.std(dim=(0, 1), unbiased=False)
    valid_modulation = target_std > 1e-8
    modulation_ratio = torch.full_like(target_std, torch.nan)
    modulation_ratio[valid_modulation] = prediction_std[valid_modulation] / target_std[valid_modulation]
    modulation_summary = finite_summary(modulation_ratio)

    causal_max_errors = []
    causal_mean_errors = []
    stream_max_errors = []
    diagnostic_length = min(32, video.shape[1])
    available = max(1, video.shape[1] - diagnostic_length + 1)
    diagnostic_starts = np.linspace(0, available - 1, diagnostic_batches, dtype=int)
    for start in diagnostic_starts:
        segment = video[:, start : start + diagnostic_length]
        full_log_rate = model(segment).clamp(max=10.0)
        segmented_stream = streaming_predict(model, segment, chunk_pattern).clamp(max=10.0)
        stream_max_errors.append(torch.max(torch.abs(segmented_stream - full_log_rate)).item())
        future_start = max(1, segment.shape[1] // 2)
        perturbed = segment.clone()
        perturbed[:, future_start:] = torch.flip(segment[:, future_start:], dims=(1,)) + 0.123
        perturbed_log_rate = model(perturbed).clamp(max=10.0)
        causal_difference = torch.abs(full_log_rate[:, :future_start] - perturbed_log_rate[:, :future_start])
        causal_max_errors.append(causal_difference.max().item())
        causal_mean_errors.append(causal_difference.mean().item())

    model.train()
    valid_response = torch.isfinite(response_correlations)
    valid_delta = torch.isfinite(delta_correlations)
    return {
        "poisson": per_neuron_poisson.mean().item(),
        "baseline_poisson": per_neuron_baseline.mean().item(),
        "response_correlation": response_summary["mean"],
        "response_correlation_median": response_summary["median"],
        "response_correlation_p25": response_summary["p25"],
        "delta_correlation": delta_summary["mean"],
        "delta_correlation_median": delta_summary["median"],
        "positive_response_fraction": (
            (response_correlations[valid_response] > 0).float().mean().item() if torch.any(valid_response) else float("nan")
        ),
        "positive_delta_fraction": (
            (delta_correlations[valid_delta] > 0).float().mean().item() if torch.any(valid_delta) else float("nan")
        ),
        "poisson_improved_fraction": (per_neuron_poisson < per_neuron_baseline).float().mean().item(),
        "modulation_ratio_mean": modulation_summary["mean"],
        "modulation_ratio_median": modulation_summary["median"],
        "modulated_neuron_fraction": (
            (modulation_ratio[valid_modulation] > 0.1).float().mean().item()
            if torch.any(valid_modulation) else float("nan")
        ),
        "valid_response_neurons": response_summary["valid"],
        "valid_delta_neurons": delta_summary["valid"],
        "causal_max_error": max(causal_max_errors, default=float("nan")),
        "causal_mean_error": float(np.mean(causal_mean_errors)) if causal_mean_errors else float("nan"),
        "stream_max_error": max(stream_max_errors, default=float("nan")),
        "unique_time_steps": int(target.shape[1]),
        "per_neuron_response_correlation": response_correlations.cpu().tolist(),
        "per_neuron_delta_correlation": delta_correlations.cpu().tolist(),
        "per_neuron_poisson": per_neuron_poisson.cpu().tolist(),
        "per_neuron_baseline_poisson": per_neuron_baseline.cpu().tolist(),
        "per_neuron_modulation_ratio": modulation_ratio.cpu().tolist(),
    }


def print_evaluation(label: str, step: int | None, metrics: dict[str, float | int]) -> None:
    prefix = label if step is None else f"step={step:05d} {label}"
    print(
        f"{prefix}_poisson={metrics['poisson']:.6f} "
        f"mean_baseline={metrics['baseline_poisson']:.6f} "
        f"better={metrics['poisson'] < metrics['baseline_poisson']} "
        f"response_r_mean={metrics['response_correlation']:.4f} "
        f"response_r_median={metrics['response_correlation_median']:.4f} "
        f"response_r_positive={metrics['positive_response_fraction']:.1%} "
        f"delta_r_mean={metrics['delta_correlation']:.4f} "
        f"delta_r_positive={metrics['positive_delta_fraction']:.1%} "
        f"neurons_better={metrics['poisson_improved_fraction']:.1%} "
        f"modulation={metrics['modulation_ratio_median']:.3f} "
        f"neurons_modulated={metrics['modulated_neuron_fraction']:.1%} "
        f"time_steps={metrics['unique_time_steps']} "
        f"causal_max={metrics['causal_max_error']:.3e} "
        f"stream_max={metrics['stream_max_error']:.3e}"
    )


def save_evaluation(path: Path, label: str, step: int | None, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    destination = path.with_name(f"{path.stem}_{label}_metrics.json")
    destination.write_text(
        json.dumps({"split": label, "step": step, **metrics}, indent=2),
        encoding="utf-8",
    )


def save_checkpoint(path, model, optimizer, config, step, args, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "steps": step,
            "args": vars(args),
            "data_manifest": manifest,
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != "Goldin et al. 2022":
        raise ValueError("The selected directory is not prepared Goldin 2022 data")
    device = choose_device(args.device)
    config = SynchronousControlConfig(
        image_size=max(int(manifest["image_height"]), int(manifest["image_width"])),
        input_channels=int(manifest["input_channels"]),
        response_dim=int(manifest["response_dim"]),
        frames_per_step=1,
        token_dim=args.token_dim,
        spatial_grid=args.spatial_grid,
        state_tokens=tuple(args.state_tokens),
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )
    model = SynchronousControlProcessor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved_config = checkpoint.get("config")
        if saved_config is not None and vars(saved_config) != vars(config):
            raise ValueError(
                "Checkpoint config differs from the current depth-sensitive fusion config. "
                "Start a new training run to apply the new fusion initialization."
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Command-line fine-tuning settings must override values stored in the
        # previous optimizer state.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.lr
            parameter_group["weight_decay"] = args.weight_decay
        step = int(checkpoint.get("steps", 0))
        print(f"resumed {args.resume} at step={step} lr={args.lr:g}")

    mean_response = torch.tensor(manifest["train_mean_response"], device=device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} parameters={parameters:,} neurons={config.response_dim} "
        f"pipeline_layers={len(config.state_tokens)} windows={manifest['windows']}"
    )
    print(
        "fusion_profile="
        f"acceptance:{config.fusion_acceptance_start:.2f}->{config.fusion_acceptance_end:.2f} "
        f"threshold:{config.fusion_threshold_start:.3f}->{config.fusion_threshold_end:.3f} "
        f"temperature:{config.fusion_temperature:.3f}"
    )
    model.train()
    running_loss = 0.0
    running_poisson = 0.0
    running_correlation = 0.0
    running_delta = 0.0
    running_count = 0
    try:
        while step < args.steps:
            for video, target in batches(args.data / "train", args.batch_size, True):
                video, target = video.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                log_rate = model(video).clamp(max=10.0)
                loss_poisson = weighted_poisson_loss(log_rate, target, args.event_weight)
                loss_correlation = temporal_correlation_loss(log_rate, target)
                loss_delta = temporal_delta_loss(log_rate, target)
                loss = (
                    loss_poisson
                    + args.correlation_weight * loss_correlation
                    + args.delta_weight * loss_delta
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                step += 1
                running_loss += loss.item()
                running_poisson += loss_poisson.item()
                running_correlation += loss_correlation.item()
                running_delta += loss_delta.item()
                running_count += 1

                if step % args.log_every == 0:
                    print(
                        f"step={step:05d} train_loss={running_loss/running_count:.6f} "
                        f"poisson={running_poisson/running_count:.6f} "
                        f"corr_loss={running_correlation/running_count:.6f} "
                        f"delta_loss={running_delta/running_count:.6f}"
                    )
                    running_loss = 0.0
                    running_poisson = 0.0
                    running_correlation = 0.0
                    running_delta = 0.0
                    running_count = 0
                if step % args.eval_every == 0:
                    validation_metrics = evaluate(
                        model, args.data / "validation", device, mean_response,
                        tuple(args.stream_chunks), args.diagnostic_batches,
                        int(manifest["stride"]), args.evaluation_max_steps,
                    )
                    print_evaluation("val", step, validation_metrics)
                    save_evaluation(args.checkpoint, "validation", step, validation_metrics)
                if args.save_every and step % args.save_every == 0:
                    save_checkpoint(args.checkpoint, model, optimizer, config, step, args, manifest)
                    print(f"saved checkpoint to {args.checkpoint}")
                if step >= args.steps:
                    break
    except KeyboardInterrupt:
        save_checkpoint(args.checkpoint, model, optimizer, config, step, args, manifest)
        print(f"interrupted; saved step={step} to {args.checkpoint}")
        return

    test_metrics = evaluate(
        model, args.data / "test", device, mean_response, tuple(args.stream_chunks),
        args.diagnostic_batches, int(manifest["stride"]), args.evaluation_max_steps,
    )
    print_evaluation("test", None, test_metrics)
    save_evaluation(args.checkpoint, "test", step, test_metrics)
    save_checkpoint(args.checkpoint, model, optimizer, config, step, args, manifest)
    print(f"finished; saved {args.checkpoint}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one Goldin 2022 retinal session")
    parser.add_argument("--data", type=Path, default=Path("data/goldin2022/processed"))
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
    parser.add_argument("--correlation-weight", type=float, default=0.25)
    parser.add_argument("--delta-weight", type=float, default=0.05)
    parser.add_argument("--event-weight", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/goldin2022_16layer.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument(
        "--validation-batches", type=int, default=8,
        help="Deprecated compatibility option; evaluation now uses unique continuous time points",
    )
    parser.add_argument(
        "--evaluation-max-steps", type=int, default=0,
        help="Limit continuous evaluation steps; 0 evaluates the complete reconstructed split",
    )
    parser.add_argument(
        "--stream-chunks", type=int, nargs="+", default=[8, 16, 7, 13],
        help="Irregular chunk sizes used for stateful streaming evaluation",
    )
    parser.add_argument(
        "--diagnostic-batches", type=int, default=2,
        help="Continuous segments used for causality and full-vs-stream checks",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.stream_chunks or any(size <= 0 for size in args.stream_chunks):
        raise SystemExit("--stream-chunks must contain positive integers")
    if args.diagnostic_batches <= 0:
        raise SystemExit("--diagnostic-batches must be positive")
    if args.evaluation_max_steps < 0:
        raise SystemExit("--evaluation-max-steps cannot be negative")
    if args.correlation_weight < 0 or args.delta_weight < 0 or args.event_weight < 0:
        raise SystemExit("Loss weights cannot be negative")
    train(args)


if __name__ == "__main__":
    main()
