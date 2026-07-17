"""Colab Gradio demo for a trained Goldin 2022 streaming checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn.functional as F

from .model import SynchronousControlProcessor
from .train import choose_device
from .train_goldin2022 import load_continuous_split, neuron_correlations, streaming_predict


def parse_chunks(value: str) -> tuple[int, ...]:
    try:
        chunks = tuple(int(item.strip()) for item in value.replace("，", ",").split(",") if item.strip())
    except ValueError as error:
        raise gr.Error("Chunk 序列必须是逗号分隔的正整数，例如 8,16,7,13") from error
    if not chunks or any(chunk <= 0 for chunk in chunks):
        raise gr.Error("Chunk 序列必须包含正整数")
    return chunks


def display_frame(frame: torch.Tensor, mean: float, std: float) -> np.ndarray:
    image = frame.permute(1, 2, 0).float().cpu().numpy() * std + mean
    low, high = np.percentile(image, (1, 99))
    image = np.clip((image - low) / max(high - low, 1e-8), 0.0, 1.0)
    if image.shape[-1] == 1:
        image = image[..., 0]
    return (image * 255).astype(np.uint8)


class GoldinDemo:
    def __init__(self, checkpoint_path: Path, data_path: Path, device_name: str) -> None:
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.device = choose_device(device_name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.manifest = checkpoint.get("data_manifest")
        if self.manifest is None:
            import json
            self.manifest = json.loads((data_path / "manifest.json").read_text(encoding="utf-8"))
        self.model = SynchronousControlProcessor(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.step = int(checkpoint.get("steps", 0))
        self.mean_response = torch.tensor(self.manifest["train_mean_response"], dtype=torch.float32)
        self.sequences = {
            split: load_continuous_split(data_path / split, int(self.manifest["stride"]))
            for split in ("validation", "test")
        }

    @property
    def neuron_count(self) -> int:
        return int(self.config.response_dim)

    def model_summary(self) -> str:
        mode = getattr(
            self.config,
            "fusion_mode",
            "depth_sensitive" if hasattr(self.config, "fusion_acceptance_start") else "legacy",
        )
        return (
            f"**Checkpoint:** `{self.checkpoint_path.name}` · **step:** {self.step} · "
            f"**device:** `{self.device}` · **neurons:** {self.neuron_count} · "
            f"**token dim:** {self.config.token_dim} · **pipeline:** {len(self.config.state_tokens)} layers · "
            f"**fusion:** `{mode}`"
        )

    @torch.inference_mode()
    def run(self, split: str, start: int, length: int, neuron: str, warmup: int, chunks_text: str):
        chunks = parse_chunks(chunks_text)
        video_cpu, target_cpu = self.sequences[split]
        total = video_cpu.shape[1]
        length = max(8, min(int(length), total))
        start = max(0, min(int(start), total - length))
        warm_start = max(0, start - int(warmup))
        stop = start + length
        video = video_cpu[:, warm_start:stop].to(self.device)
        target = target_cpu[:, start:stop].to(self.device)
        offset = start - warm_start

        streamed_all = streaming_predict(self.model, video, chunks).clamp(max=10.0)
        full_all = self.model(video).clamp(max=10.0)
        streamed = streamed_all[:, offset:]
        rate = torch.exp(streamed)
        baseline_rate = self.mean_response.to(self.device).view(1, 1, -1).expand_as(target)

        response_r = neuron_correlations(rate, target)
        delta_r = neuron_correlations(torch.diff(rate, dim=1), torch.diff(target, dim=1))
        model_nll = F.poisson_nll_loss(
            streamed, target, log_input=True, full=False, reduction="none"
        ).mean(dim=(0, 1))
        baseline_nll = F.poisson_nll_loss(
            torch.log(baseline_rate.clamp_min(1e-6)), target,
            log_input=True, full=False, reduction="none",
        ).mean(dim=(0, 1))
        target_std = target.std(dim=(0, 1), unbiased=False)
        prediction_std = rate.std(dim=(0, 1), unbiased=False)
        modulation_ratio = prediction_std / target_std.clamp_min(1e-8)

        cutoff = max(1, video.shape[1] - length // 2)
        perturbed = video.clone()
        perturbed[:, cutoff:] = torch.flip(video[:, cutoff:], dims=(1,)) + 0.123
        changed = self.model(perturbed).clamp(max=10.0)
        causal_max = torch.max(torch.abs(full_all[:, :cutoff] - changed[:, :cutoff])).item()
        stream_max = torch.max(torch.abs(streamed_all - full_all)).item()

        neuron_index = max(0, min(int(neuron), self.neuron_count - 1))
        x = np.arange(start, stop)
        selected_target = target[0, :, neuron_index].cpu().numpy()
        selected_rate = rate[0, :, neuron_index].cpu().numpy()
        selected_baseline = baseline_rate[0, :, neuron_index].cpu().numpy()
        line = go.Figure()
        line.add_trace(go.Scatter(x=x, y=selected_target, name="真实响应", line=dict(color="#21c78b", width=2)))
        line.add_trace(go.Scatter(x=x, y=selected_rate, name="流式模型", line=dict(color="#6d68ff", width=2)))
        line.add_trace(go.Scatter(x=x, y=selected_baseline, name="均值基线", line=dict(color="#f59e0b", dash="dot")))
        line.update_layout(
            title=f"神经元 {neuron_index} · 连续响应", xaxis_title="唯一时间点", yaxis_title="响应率",
            template="plotly_white", height=390, margin=dict(l=50, r=20, t=55, b=45),
            hovermode="x unified",
        )

        heatmap = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("真实响应", "流式模型响应"))
        target_matrix = target[0].transpose(0, 1).cpu().numpy()
        rate_matrix = rate[0].transpose(0, 1).cpu().numpy()
        color_max = max(
            float(np.percentile(np.concatenate((target_matrix.ravel(), rate_matrix.ravel())), 99)),
            1e-6,
        )
        heatmap.add_trace(go.Heatmap(z=target_matrix, x=x, colorscale="Viridis", zmin=0, zmax=color_max, showscale=False), row=1, col=1)
        heatmap.add_trace(go.Heatmap(z=rate_matrix, x=x, colorscale="Viridis", zmin=0, zmax=color_max), row=2, col=1)
        heatmap.update_yaxes(title_text="神经元", row=1, col=1)
        heatmap.update_yaxes(title_text="神经元", row=2, col=1)
        heatmap.update_xaxes(title_text="唯一时间点", row=2, col=1)
        heatmap.update_layout(template="plotly_white", height=570, margin=dict(l=55, r=25, t=65, b=45))

        finite_r = response_r[torch.isfinite(response_r)]
        finite_delta = delta_r[torch.isfinite(delta_r)]
        summary = (
            f"### 本段流式评估\n"
            f"- 区间：`{split}[{start}:{stop}]`，预热 `{offset}` 步，chunk `{chunks}`\n"
            f"- Poisson：模型 `{model_nll.mean().item():.6f}` / 基线 `{baseline_nll.mean().item():.6f}`\n"
            f"- response r：均值 `{finite_r.mean().item() if len(finite_r) else float('nan'):.4f}`，"
            f"正相关 `{((finite_r > 0).float().mean().item() if len(finite_r) else float('nan')):.1%}`\n"
            f"- delta r：均值 `{finite_delta.mean().item() if len(finite_delta) else float('nan'):.4f}`\n"
            f"- modulation ratio：中位数 `{modulation_ratio.median().item():.4f}`，"
            f"超过 0.1 的神经元 `{(modulation_ratio > 0.1).float().mean().item():.1%}`\n"
            f"- 改善神经元：`{(model_nll < baseline_nll).float().mean().item():.1%}`\n"
            f"- 因果误差：`{causal_max:.3e}`；流式一致性误差：`{stream_max:.3e}`"
        )
        rows = [
            [
                index,
                float(response_r[index].cpu()),
                float(delta_r[index].cpu()),
                float(model_nll[index].cpu()),
                float(baseline_nll[index].cpu()),
                float(modulation_ratio[index].cpu()),
                bool(model_nll[index] < baseline_nll[index]),
            ]
            for index in range(self.neuron_count)
        ]
        frame = display_frame(
            video_cpu[0, start], float(self.manifest["stimulus_mean"]), float(self.manifest["stimulus_std"])
        )
        return frame, line, heatmap, summary, rows


def build_app(runtime: GoldinDemo) -> gr.Blocks:
    maximum_steps = max(sequence[0].shape[1] for sequence in runtime.sequences.values())
    with gr.Blocks(title="AutoMachine · Goldin 2022 流式神经响应") as app:
        gr.Markdown("# AutoMachine · 视觉刺激 → 视网膜响应")
        gr.Markdown(runtime.model_summary())
        with gr.Row():
            split = gr.Dropdown(("validation", "test"), value="validation", label="数据分段")
            start = gr.Slider(0, max(0, maximum_steps - 8), value=0, step=1, label="起始时间点")
            length = gr.Slider(8, min(256, maximum_steps), value=min(64, maximum_steps), step=1, label="展示长度")
        with gr.Row():
            neuron = gr.Dropdown([str(i) for i in range(runtime.neuron_count)], value="0", label="重点神经元")
            warmup = gr.Slider(0, 256, value=64, step=1, label="流式状态预热步数")
            chunks = gr.Textbox(value="8,16,7,13", label="流式 chunk 序列")
            run = gr.Button("运行流式演示", variant="primary")
        with gr.Row():
            frame = gr.Image(label="当前视觉刺激", height=320)
            summary = gr.Markdown()
        response_plot = gr.Plot(label="选定神经元响应")
        population_plot = gr.Plot(label="全神经元时间响应")
        table = gr.Dataframe(
            headers=("neuron", "response_r", "delta_r", "model_nll", "baseline_nll", "modulation", "improved"),
            datatype=("number", "number", "number", "number", "number", "number", "bool"),
            label="逐神经元指标", interactive=False,
        )
        run.click(
            runtime.run,
            inputs=(split, start, length, neuron, warmup, chunks),
            outputs=(frame, response_plot, population_plot, summary, table),
        )
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Goldin 2022 streaming Gradio demo")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/goldin2022_depth_fusion_v2.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/goldin2022/processed"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = GoldinDemo(args.checkpoint, args.data, args.device)
    build_app(runtime).launch(
        server_name=args.host, server_port=args.port, share=args.share, debug=True,
    )


if __name__ == "__main__":
    main()
