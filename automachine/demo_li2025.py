"""Interactive Colab demo for Li 2025 continuous natural-movie responses."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from .model import SynchronousControlProcessor
from .train import choose_device
from .train_goldin2022 import neuron_correlations, streaming_predict
from .train_li2025 import load_episodes


class LiDemo:
    def __init__(self, checkpoint: Path, data: Path, device: str) -> None:
        self.device = choose_device(device)
        saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = SynchronousControlProcessor(saved["config"]).to(self.device)
        self.model.load_state_dict(saved["model"]); self.model.eval()
        self.config = saved["config"]; self.step = int(saved.get("steps", 0)); self.checkpoint = checkpoint
        self.sequences = {split: load_episodes(data / split)[0] for split in ("validation", "test")}

    @torch.inference_mode()
    def run(self, split, start, length, neuron, warmup, chunks_text):
        chunks = tuple(int(item.strip()) for item in chunks_text.split(",") if item.strip())
        if not chunks or any(item <= 0 for item in chunks): raise gr.Error("chunk 必须是正整数，例如 8,13,7")
        video_cpu, target_cpu = self.sequences[split]
        total = video_cpu.shape[1]; length = max(8, min(int(length), total)); start = max(0, min(int(start), total - length))
        warm_start = max(0, start - int(warmup)); stop = start + length; offset = start - warm_start
        video = video_cpu[:, warm_start:stop].to(self.device); target = target_cpu[:, start:stop].to(self.device)
        prediction_all = streaming_predict(self.model, video, chunks); prediction = prediction_all[:, offset:]
        response_r = neuron_correlations(prediction, target)
        delta_r = neuron_correlations(torch.diff(prediction, dim=1), torch.diff(target, dim=1))
        model_mse = (prediction - target).square().mean((0, 1)); baseline_mse = target.square().mean((0, 1))
        full = self.model(video); stream_error = (full - prediction_all).abs().max().item()
        cutoff = max(1, video.shape[1] // 2); changed = video.clone(); changed[:, cutoff:] = torch.flip(changed[:, cutoff:], (1,))
        causal_error = (full[:, :cutoff] - self.model(changed)[:, :cutoff]).abs().max().item()
        index = max(0, min(int(neuron), self.config.response_dim - 1)); x = np.arange(start, stop)
        line = go.Figure()
        line.add_scatter(x=x, y=target[0, :, index].cpu(), name="真实钙响应", line=dict(color="#21c78b"))
        line.add_scatter(x=x, y=prediction[0, :, index].cpu(), name="流式模型", line=dict(color="#6d68ff"))
        line.add_scatter(x=x, y=np.zeros(len(x)), name="训练均值基线", line=dict(color="#f59e0b", dash="dot"))
        line.update_layout(title=f"神经元 {index} · 连续自然电影响应", xaxis_title="响应时间点（约 5 Hz）", yaxis_title="标准化钙响应", hovermode="x unified", template="plotly_white")
        heatmap = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("真实响应", "流式模型响应"))
        heatmap.add_trace(go.Heatmap(z=target[0].T.cpu(), x=x, colorscale="RdBu", zmid=0, showscale=False), 1, 1)
        heatmap.add_trace(go.Heatmap(z=prediction[0].T.cpu(), x=x, colorscale="RdBu", zmid=0), 2, 1)
        heatmap.update_layout(height=600, template="plotly_white")
        finite = response_r[torch.isfinite(response_r)]; finite_delta = delta_r[torch.isfinite(delta_r)]
        summary = (f"### 连续流式评估\n- 区间：`{split}[{start}:{stop}]`，预热 `{offset}` 步\n"
                   f"- MSE：模型 `{model_mse.mean():.5f}` / 均值基线 `{baseline_mse.mean():.5f}`\n"
                   f"- response r：`{finite.mean() if len(finite) else float('nan'):.4f}`；delta r：`{finite_delta.mean() if len(finite_delta) else float('nan'):.4f}`\n"
                   f"- 改善神经元：`{(model_mse < baseline_mse).float().mean():.1%}`\n- 因果误差：`{causal_error:.2e}`；流式一致性：`{stream_error:.2e}`")
        frame = (video_cpu[0, start].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        if frame.shape[-1] == 1: frame = frame[..., 0]
        return frame, line, heatmap, summary


def build_app(runtime: LiDemo) -> gr.Blocks:
    maximum = max(item[0].shape[1] for item in runtime.sequences.values())
    with gr.Blocks(title="AutoMachine · Li 2025") as app:
        gr.Markdown(f"# 自然电影 → 小鼠上丘连续钙响应\n`{runtime.checkpoint.name}` · step {runtime.step} · 16 层因果 Token 管线")
        with gr.Row():
            split = gr.Dropdown(("validation", "test"), value="validation", label="完整电影分段")
            start = gr.Slider(0, max(0, maximum - 8), 0, step=1, label="拖动时间")
            length = gr.Slider(8, min(256, maximum), min(64, maximum), step=1, label="显示长度")
        with gr.Row():
            neuron = gr.Dropdown([str(i) for i in range(runtime.config.response_dim)], value="0", label="神经元")
            warmup = gr.Slider(0, 256, 64, step=1, label="状态预热")
            chunks = gr.Textbox("8,13,7", label="流式 chunk")
            refresh = gr.Button("刷新", variant="primary")
        frame = gr.Image(label="当前电影帧", height=300); summary = gr.Markdown(); line = gr.Plot(); heatmap = gr.Plot()
        inputs, outputs = (split, start, length, neuron, warmup, chunks), (frame, line, heatmap, summary)
        refresh.click(runtime.run, inputs, outputs); start.release(runtime.run, inputs, outputs); neuron.change(runtime.run, inputs, outputs); split.change(runtime.run, inputs, outputs)
        app.load(runtime.run, inputs, outputs)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/li2025_sc_16layer.pt")); parser.add_argument("--data", type=Path, default=Path("data/li2025/plane_one"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); parser.add_argument("--port", type=int, default=7860); parser.add_argument("--share", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(); build_app(LiDemo(args.checkpoint, args.data, args.device)).launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__": main()
