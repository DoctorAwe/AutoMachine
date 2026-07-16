"""Gradio dashboard for inspecting trained feature-stream checkpoints in Colab."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from .feature_model import FeatureStreamProcessor, FeatureStreamProcessorConfig


DEFAULT_NAMES = (
    "Body Acc X", "Body Acc Y", "Body Acc Z",
    "Body Gyro X", "Body Gyro Y", "Body Gyro Z",
    "Total Acc X", "Total Acc Y", "Total Acc Z",
)


def _checkpoint_config(raw: Any) -> FeatureStreamProcessorConfig:
    if isinstance(raw, FeatureStreamProcessorConfig):
        return raw
    if isinstance(raw, dict):
        allowed = FeatureStreamProcessorConfig.__dataclass_fields__.keys()
        return FeatureStreamProcessorConfig(**{key: raw[key] for key in allowed if key in raw})
    raise TypeError(f"Unsupported checkpoint config: {type(raw).__name__}")


@lru_cache(maxsize=8)
def load_model(path: str, device_name: str) -> tuple[FeatureStreamProcessor, FeatureStreamProcessorConfig]:
    device = torch.device(device_name)
    # Checkpoints are produced locally by this project and contain a dataclass.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = _checkpoint_config(checkpoint["config"])
    model = FeatureStreamProcessor(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def load_dataset(path: Path) -> tuple[np.ndarray, list[str], np.ndarray | None, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as data:
        sequences = np.asarray(data["sequences"], dtype=np.float32)
        names = [str(x) for x in data.get("feature_names", np.asarray(DEFAULT_NAMES))]
        labels = np.asarray(data["labels"]) if "labels" in data else None
        subjects = np.asarray(data["subjects"]) if "subjects" in data else None
    return sequences, names, labels, subjects


def predict_sample(
    checkpoint_path: str,
    sample_index: int,
    sequences: np.ndarray,
    feature_names: list[str],
    labels: np.ndarray | None,
    subjects: np.ndarray | None,
    device_name: str,
) -> tuple[go.Figure, str, list[list[str]]]:
    model, config = load_model(checkpoint_path, device_name)
    sample_index = int(np.clip(sample_index, 0, len(sequences) - 1))
    sequence = sequences[sample_index]
    input_length = min(len(sequence) - 1, 32 if len(sequence) > 32 else len(sequence) - 1)
    # Use the length the small validation run was trained with when available.
    if len(sequence) >= 64 and "stream" in Path(checkpoint_path).stem.lower():
        input_length = 64
    inputs = sequence[:input_length]

    device = next(model.parameters()).device
    with torch.inference_mode():
        prediction = model(torch.from_numpy(inputs).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

    chunk = config.chunk_size
    target = sequence[chunk : chunk + len(prediction)]
    prediction = prediction[: len(target)]
    persistence = inputs[chunk - 1 : chunk - 1 + len(target)]
    x_input = np.arange(input_length)
    x_output = np.arange(chunk, chunk + len(target))

    channels = min(inputs.shape[-1], 9)
    rows = int(np.ceil(channels / 3))
    titles = feature_names[:channels]
    figure = make_subplots(rows=rows, cols=3, subplot_titles=titles, shared_xaxes=True)
    for channel in range(channels):
        row, col = divmod(channel, 3)
        show_legend = channel == 0
        figure.add_trace(
            go.Scatter(x=x_input, y=inputs[:, channel], name="输入", mode="lines", line={"color": "#64748b"}, showlegend=show_legend),
            row=row + 1, col=col + 1,
        )
        figure.add_trace(
            go.Scatter(x=x_output, y=target[:, channel], name="真实", mode="lines", line={"color": "#10b981", "width": 2}, showlegend=show_legend),
            row=row + 1, col=col + 1,
        )
        figure.add_trace(
            go.Scatter(x=x_output, y=prediction[:, channel], name="模型预测", mode="lines", line={"color": "#6366f1", "width": 2}, showlegend=show_legend),
            row=row + 1, col=col + 1,
        )
        figure.add_trace(
            go.Scatter(x=x_output, y=persistence[:, channel], name="复制基线", mode="lines", line={"color": "#f59e0b", "dash": "dot"}, showlegend=show_legend),
            row=row + 1, col=col + 1,
        )
    figure.update_layout(
        height=760,
        margin={"l": 42, "r": 24, "t": 70, "b": 45},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        template="plotly_white",
    )
    figure.update_xaxes(title_text="采样点")

    model_mse_channels = ((prediction - target) ** 2).mean(axis=0)
    base_mse_channels = ((persistence - target) ** 2).mean(axis=0)
    model_mse = float(model_mse_channels.mean())
    model_mae = float(np.abs(prediction - target).mean())
    base_mse = float(base_mse_channels.mean())
    improvement = 100.0 * (base_mse - model_mse) / max(base_mse, 1e-12)
    activity = int(labels[sample_index]) + 1 if labels is not None else "未知"
    subject = int(subjects[sample_index]) if subjects is not None else "未知"
    summary = (
        f"**样本 {sample_index}** · 受试者 `{subject}` · 活动标签 `{activity}`  \n"
        f"模型 MSE **{model_mse:.4f}** · MAE **{model_mae:.4f}** · "
        f"复制基线 MSE **{base_mse:.4f}** · 相对改善 **{improvement:+.1f}%**"
    )
    table = [
        [feature_names[i], f"{model_mse_channels[i]:.5f}", f"{base_mse_channels[i]:.5f}"]
        for i in range(channels)
    ]
    return figure, summary, table


def build_app(data_path: Path, checkpoint_dir: Path, device_name: str) -> gr.Blocks:
    sequences, names, labels, subjects = load_dataset(data_path)
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoints found in {checkpoint_dir}")
    choices = [str(path) for path in checkpoints]

    def run(checkpoint: str, sample: int):
        return predict_sample(checkpoint, sample, sequences, names, labels, subjects, device_name)

    css = """
    .dashboard-title { text-align:center; margin-bottom: 0.25rem; }
    .dashboard-subtitle { text-align:center; color:#64748b; margin-bottom:1rem; }
    """
    with gr.Blocks(title="AutoMachine 流式模型实验台", css=css) as app:
        gr.Markdown("# AutoMachine 流式模型实验台", elem_classes="dashboard-title")
        gr.Markdown("UCI HAR 九通道输入、真实后续信号、模型预测与持久性基线对比", elem_classes="dashboard-subtitle")
        with gr.Row():
            checkpoint = gr.Dropdown(choices=choices, value=choices[0], label="模型 checkpoint")
            sample = gr.Slider(0, len(sequences) - 1, value=0, step=1, label="测试样本")
            run_button = gr.Button("运行预测", variant="primary")
        summary = gr.Markdown()
        plot = gr.Plot(label="九通道时序对比")
        metrics = gr.Dataframe(
            headers=["通道", "模型 MSE", "复制基线 MSE"],
            datatype=["str", "str", "str"],
            interactive=False,
            label="逐通道误差",
        )
        run_button.click(run, inputs=[checkpoint, sample], outputs=[plot, summary, metrics])
        checkpoint.change(run, inputs=[checkpoint, sample], outputs=[plot, summary, metrics])
        sample.release(run, inputs=[checkpoint, sample], outputs=[plot, summary, metrics])
        app.load(run, inputs=[checkpoint, sample], outputs=[plot, summary, metrics])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the AutoMachine model dashboard")
    parser.add_argument("--data", type=Path, default=Path("data/uci_har/test.npz"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--share", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    app = build_app(args.data, args.checkpoints, device)
    app.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
