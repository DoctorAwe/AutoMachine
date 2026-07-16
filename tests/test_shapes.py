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


def test_video_encoder_is_sensitive_to_frame_order() -> None:
    torch.manual_seed(3)
    config = NeuralStreamProcessorConfig(
        image_size=32, chunk_size=3, token_dim=32, encoder_grid=4,
        state_tokens=(8, 12), num_heads=4,
    )
    model = NeuralStreamProcessor(config).eval()
    video = torch.rand(1, 3, 3, 32, 32)

    forward, _ = model.forward_step(video)
    reverse, _ = model.forward_step(video.flip(1))

    assert not torch.allclose(forward, reverse)


def test_all_memory_scales_receive_decoder_gradient() -> None:
    config = NeuralStreamProcessorConfig(
        image_size=32, chunk_size=2, token_dim=32, encoder_grid=4,
        state_tokens=(8, 12, 16), num_heads=4,
    )
    model = NeuralStreamProcessor(config)
    model(torch.rand(1, 2, 3, 32, 32)).mean().backward()

    for layer in model.memory.layers:
        assert layer.gate.weight.grad is not None
        assert torch.isfinite(layer.gate.weight.grad).all()
