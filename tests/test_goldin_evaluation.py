from __future__ import annotations

import torch
import numpy as np

from automachine import SynchronousControlConfig, SynchronousControlProcessor
from automachine.train_goldin2022 import (
    load_continuous_split,
    mean_neuron_correlation,
    neuron_correlations,
    streaming_predict,
    temporal_correlation_loss,
)


def test_streaming_evaluation_matches_full_causal_output() -> None:
    torch.manual_seed(11)
    config = SynchronousControlConfig(
        image_size=16,
        input_channels=1,
        response_dim=3,
        token_dim=32,
        spatial_grid=2,
        state_tokens=(4, 6, 8),
        num_heads=4,
    )
    model = SynchronousControlProcessor(config).eval()
    video = torch.randn(2, 9, 1, 16, 16)
    with torch.inference_mode():
        full = model(video)
        streamed = streaming_predict(model, video, (1, 3, 2))
    torch.testing.assert_close(streamed, full, rtol=1e-5, atol=1e-6)


def test_interval_response_correlation_is_neuronwise() -> None:
    target = torch.tensor([[[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]])
    correlation, valid_neurons = mean_neuron_correlation(target * 2.0 + 1.0, target)
    assert valid_neurons == 2
    assert abs(correlation - 1.0) < 1e-6


def test_continuous_evaluation_removes_overlapping_time_points(tmp_path) -> None:
    split = tmp_path / "validation"
    split.mkdir()
    sequence = np.arange(5, dtype=np.float32)
    frames = np.stack([sequence[0:3], sequence[2:5]])[..., None, None, None]
    responses = np.stack([sequence[0:3], sequence[2:5]])[..., None]
    np.savez(split / "shard_0000.npz", frames=frames, responses=responses, starts=np.array([0, 2]))

    video, target = load_continuous_split(split, stride=2)
    assert video.shape == (1, 5, 1, 1, 1)
    torch.testing.assert_close(video[0, :, 0, 0, 0], torch.arange(5, dtype=torch.float32))
    torch.testing.assert_close(target[0, :, 0], torch.arange(5, dtype=torch.float32))


def test_per_neuron_correlation_preserves_distribution() -> None:
    target = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    prediction = torch.stack((target[..., 0], torch.flip(target[..., 1], dims=(1,))), dim=-1)
    correlations = neuron_correlations(prediction, target)
    torch.testing.assert_close(correlations, torch.tensor([1.0, -1.0]))


def test_correlation_loss_penalizes_flat_rate_output() -> None:
    target = torch.tensor([[[0.0], [1.0], [2.0], [4.0]]])
    flat_log_rate = torch.zeros_like(target, requires_grad=True)
    matching_log_rate = torch.log(target + 0.1).requires_grad_()
    flat_loss = temporal_correlation_loss(flat_log_rate, target)
    matching_loss = temporal_correlation_loss(matching_log_rate, target)
    assert matching_loss < flat_loss
    flat_loss.backward()
    assert torch.isfinite(flat_log_rate.grad).all()
