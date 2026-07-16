"""Smoke test for the neural stream processor prototype.

Run:
    python -m tests.smoke_forward
"""

from __future__ import annotations

import torch

from automachine import NeuralStreamProcessor, NeuralStreamProcessorConfig


def main() -> None:
    torch.manual_seed(7)
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

    expected_shape = (2, 4, 3, 64, 64)
    if tuple(output.shape) != expected_shape:
        raise AssertionError(f"Expected {expected_shape}, got {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise AssertionError("Model output contains non-finite values")

    parameters = sum(param.numel() for param in model.parameters())
    print(f"OK output_shape={tuple(output.shape)} parameters={parameters:,}")


if __name__ == "__main__":
    main()
