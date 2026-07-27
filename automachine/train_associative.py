"""Controlled Colab experiment for fixed-capacity episodic association."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import SynchronousControlConfig, SynchronousControlProcessor
from .synthetic import EpisodicAssociationDataset
from .train import choose_device


def config_from_args(args: argparse.Namespace) -> SynchronousControlConfig:
    return SynchronousControlConfig(
        image_size=args.image_size,
        input_channels=3,
        auxiliary_features=4,
        response_dim=4,
        token_dim=args.token_dim,
        spatial_grid=2,
        state_tokens=(4,) * 16,
        num_heads=4,
        mlp_ratio=2,
        associative_memory_enabled=True,
        associative_memory_capacity=args.memory_capacity,
        associative_value_tokens=4,
        associative_top_k=4,
        associative_retrieval_threshold=args.retrieval_threshold,
        associative_retrieval_temperature=0.07,
        associative_write_threshold=0.55,
        associative_merge_key_threshold=0.97,
        associative_merge_value_threshold=0.90,
        associative_protected_fraction=0.125,
        associative_candidate_fraction=0.25,
    )


def dataset(args: argparse.Namespace, samples: int, offset: int) -> EpisodicAssociationDataset:
    return EpisodicAssociationDataset(
        samples=samples,
        image_size=args.image_size,
        cue_frames=args.cue_frames,
        gap=args.gap,
        query_frames=args.query_frames,
        seed_offset=offset,
    )


@torch.inference_mode()
def evaluate(model, loader, device, args) -> dict[str, float]:
    model.eval()
    correct_memory = correct_empty = correct_wrong = total = 0
    occupied = []
    prefix_length = 4 * args.cue_frames + args.gap
    for video, auxiliary, target, salience in loader:
        video, auxiliary = video.to(device), auxiliary.to(device)
        target, salience = target.to(device), salience.to(device)

        prefix_video, query_video = video[:, :prefix_length], video[:, prefix_length:]
        prefix_aux, query_aux = auxiliary[:, :prefix_length], auxiliary[:, prefix_length:]
        prefix_salience = salience[:, :prefix_length]

        _, remembered = model.forward_chunk(
            prefix_video, auxiliary=prefix_aux, memory_salience=prefix_salience
        )
        normal, _ = model.forward_chunk(
            query_video, auxiliary=query_aux, state=remembered,
            update_associative_memory=False,
        )
        correct_memory += (normal.mean(1).argmax(-1) == target).sum().item()
        occupied.append(model.associative_memory_statistics(remembered)["occupied"] / len(video))

        empty = model.init_state(len(video), device, video.dtype)
        empty.layers = tuple(layer.clone() for layer in remembered.layers)
        no_memory, _ = model.forward_chunk(
            query_video, auxiliary=query_aux, state=empty,
            update_associative_memory=False,
        )
        correct_empty += (no_memory.mean(1).argmax(-1) == target).sum().item()

        wrong = remembered.detach()
        assert wrong.associative is not None
        wrong.associative.values = wrong.associative.values.clone()
        for batch_index in range(len(video)):
            used = torch.flatnonzero(wrong.associative.occupied[batch_index])
            if len(used) > 1:
                wrong.associative.values[batch_index, used] = torch.roll(
                    wrong.associative.values[batch_index, used], shifts=1, dims=0
                )
        wrong_memory, _ = model.forward_chunk(
            query_video, auxiliary=query_aux, state=wrong,
            update_associative_memory=False,
        )
        correct_wrong += (wrong_memory.mean(1).argmax(-1) == target).sum().item()
        total += len(video)
    model.train()
    with_memory = correct_memory / total
    without_memory = correct_empty / total
    return {
        "recall_accuracy": with_memory,
        "without_memory_accuracy": without_memory,
        "wrong_memory_accuracy": correct_wrong / total,
        "memory_gain": with_memory - without_memory,
        "mean_occupied_slots": sum(occupied) / len(occupied),
    }


def train(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model = SynchronousControlProcessor(config_from_args(args)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(
        dataset(args, max(args.steps * args.batch_size, 1024), 0),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    validation_loader = DataLoader(
        dataset(args, args.validation_samples, 1_000_000),
        batch_size=args.batch_size, shuffle=False,
    )
    step = 0
    best_gain = float("-inf")
    model.train()
    while step < args.steps:
        for video, auxiliary, target, salience in train_loader:
            video, auxiliary = video.to(device), auxiliary.to(device)
            target, salience = target.to(device), salience.to(device)
            optimizer.zero_grad(set_to_none=True)
            learning_steps = 4 * args.cue_frames
            query_start = video.shape[1] - args.query_frames
            learning_output, state = model.forward_chunk(
                video[:, :learning_steps],
                auxiliary=auxiliary[:, :learning_steps],
                memory_salience=salience[:, :learning_steps],
            )
            # Long distractors test persistence but need no gradient history.
            state = state.detach()
            with torch.no_grad():
                _, state = model.forward_chunk(
                    video[:, learning_steps:query_start],
                    auxiliary=auxiliary[:, learning_steps:query_start],
                    state=state,
                    update_associative_memory=False,
                )
            state = state.detach()
            query_output, _ = model.forward_chunk(
                video[:, query_start:],
                auxiliary=auxiliary[:, query_start:],
                state=state,
                update_associative_memory=False,
            )
            logits = query_output.mean(dim=1)
            learning_target = auxiliary[:, :learning_steps].argmax(dim=-1)
            learning_loss = nn.functional.cross_entropy(
                learning_output.reshape(-1, 4),
                learning_target.reshape(-1),
            )
            recall_loss = nn.functional.cross_entropy(logits, target)
            loss = recall_loss + args.learning_weight * learning_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % args.log_every == 0:
                accuracy = (logits.argmax(-1) == target).float().mean().item()
                print(f"step={step:04d} loss={loss.item():.6f} train_recall={accuracy:.1%}")
            if step % args.eval_every == 0:
                metrics = evaluate(model, validation_loader, device, args)
                print(
                    f"step={step:04d} recall={metrics['recall_accuracy']:.1%} "
                    f"empty={metrics['without_memory_accuracy']:.1%} "
                    f"wrong={metrics['wrong_memory_accuracy']:.1%} "
                    f"gain={metrics['memory_gain']:.1%} "
                    f"slots={metrics['mean_occupied_slots']:.1f}/{args.memory_capacity}"
                )
                if metrics["memory_gain"] > best_gain:
                    best_gain = metrics["memory_gain"]
                    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model": model.state_dict(), "config": model.config,
                        "steps": step, "metrics": metrics, "args": vars(args),
                    }, args.checkpoint)
                    print(f"saved best checkpoint to {args.checkpoint}")
            if step >= args.steps:
                break

    metrics = evaluate(model, validation_loader, device, args)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.checkpoint.with_name(f"{args.checkpoint.stem}_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"final metrics: {json.dumps(metrics)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the controlled associative-recall task")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=24)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--cue-frames", type=int, default=2)
    parser.add_argument("--gap", type=int, default=48)
    parser.add_argument("--query-frames", type=int, default=2)
    parser.add_argument("--memory-capacity", type=int, default=32)
    parser.add_argument("--retrieval-threshold", type=float, default=0.40)
    parser.add_argument("--learning-weight", type=float, default=0.5)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/associative_smoke.pt"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gap <= 16:
        raise SystemExit("--gap must exceed the 16-stage short-term pipeline")
    train(args)


if __name__ == "__main__":
    main()
