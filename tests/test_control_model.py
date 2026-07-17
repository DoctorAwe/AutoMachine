from __future__ import annotations

import torch

from automachine import DEFAULT_STATE_TOKENS, SynchronousControlConfig, SynchronousControlProcessor


def test_default_pipeline_has_sixteen_stages_and_growing_tail() -> None:
    assert len(DEFAULT_STATE_TOKENS) == 16
    assert DEFAULT_STATE_TOKENS[-1] > DEFAULT_STATE_TOKENS[0]


def small_config(**overrides) -> SynchronousControlConfig:
    values = dict(
        image_size=32,
        response_dim=6,
        frames_per_step=2,
        token_dim=32,
        spatial_grid=2,
        state_tokens=(4, 6, 8),
        num_heads=4,
    )
    values.update(overrides)
    return SynchronousControlConfig(**values)


def test_synchronous_output_shape_and_state() -> None:
    model = SynchronousControlProcessor(small_config())
    video = torch.rand(2, 6, 3, 32, 32)

    response, state = model.forward_chunk(video)

    assert response.shape == (2, 3, 6)
    assert state.step == 3
    assert [layer.shape for layer in state.layers] == [(2, 4, 32), (2, 6, 32), (2, 8, 32)]
    assert torch.isfinite(response).all()


def test_future_frames_cannot_change_past_responses() -> None:
    torch.manual_seed(4)
    model = SynchronousControlProcessor(small_config()).eval()
    original = torch.rand(1, 8, 3, 32, 32)
    changed = original.clone()
    changed[:, 4:] = torch.rand_like(changed[:, 4:])

    first = model(original)
    second = model(changed)

    torch.testing.assert_close(first[:, :2], second[:, :2])


def test_chunked_stream_matches_whole_stream() -> None:
    torch.manual_seed(5)
    model = SynchronousControlProcessor(small_config()).eval()
    video = torch.rand(1, 8, 3, 32, 32)

    whole = model(video)
    left, state = model.forward_chunk(video[:, :3])
    right, state = model.forward_chunk(video[:, 3:], state=state)

    torch.testing.assert_close(whole, torch.cat([left, right], dim=1))
    assert state.step == 4
    assert state.pending_video is not None and state.pending_video.shape[1] == 0


def test_auxiliary_sensor_fusion() -> None:
    model = SynchronousControlProcessor(small_config(auxiliary_features=5))
    video = torch.rand(2, 6, 3, 32, 32)
    sensors = torch.rand(2, 6, 5)

    response = model(video, sensors)

    assert response.shape == (2, 3, 6)


def test_new_input_moves_only_one_token_layer_per_step() -> None:
    torch.manual_seed(6)
    model = SynchronousControlProcessor(small_config()).eval()
    first_group = torch.rand(1, 2, 3, 32, 32)
    second_a = torch.rand(1, 2, 3, 32, 32)
    second_b = torch.rand(1, 2, 3, 32, 32)
    _, shared = model.forward_chunk(first_group)
    _, state_a = model.forward_chunk(second_a, state=shared)
    _, state_b = model.forward_chunk(second_b, state=shared)

    assert not torch.equal(state_a.layers[0], state_b.layers[0])
    torch.testing.assert_close(state_a.layers[1], state_b.layers[1])
    torch.testing.assert_close(state_a.layers[2], state_b.layers[2])


def test_incomplete_group_is_buffered_across_chunks() -> None:
    model = SynchronousControlProcessor(small_config()).eval()
    one_frame = torch.rand(1, 1, 3, 32, 32)
    output, state = model.forward_chunk(one_frame)
    assert output.shape == (1, 0, 6)
    assert state.step == 0
    assert state.pending_video is not None and state.pending_video.shape[1] == 1

    output, state = model.forward_chunk(one_frame, state=state)
    assert output.shape == (1, 1, 6)
    assert state.step == 1
