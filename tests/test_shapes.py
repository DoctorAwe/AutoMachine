from __future__ import annotations

import torch

from automachine import NeuralStreamProcessor, NeuralStreamProcessorConfig


def test_forward_shape_and_state() -> None:
    config = NeuralStreamProcessorConfig(
        image_size=64,
        chunk_size=3,
        token_dim=64,
        state_tokens=(64, 64, 96),
        num_heads=4,
    )
    model = NeuralStreamProcessor(config)
    video = torch.rand(2, 6, 3, 64, 64)

    output = model(video)

    assert output.shape == (2, 4, 3, 64, 64)
    assert torch.isfinite(output).all()
