from __future__ import annotations

import torch

from automachine import FeatureStreamProcessor, FeatureStreamProcessorConfig


def test_feature_stream_shape() -> None:
    config = FeatureStreamProcessorConfig(
        input_features=10,
        output_features=10,
        chunk_size=4,
        token_dim=32,
        tokens_per_step=8,
        state_tokens=(16, 24),
        num_heads=4,
    )
    model = FeatureStreamProcessor(config)
    sequence = torch.rand(2, 12, 10)

    output = model(sequence)

    assert output.shape == (2, 9, 10)
    assert torch.isfinite(output).all()
