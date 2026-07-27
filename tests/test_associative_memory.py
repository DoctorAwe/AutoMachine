from __future__ import annotations

import torch

from automachine import SynchronousControlConfig, SynchronousControlProcessor


def memory_config() -> SynchronousControlConfig:
    return SynchronousControlConfig(
        image_size=16,
        response_dim=4,
        token_dim=32,
        spatial_grid=2,
        state_tokens=(4, 4),
        num_heads=4,
        associative_memory_capacity=8,
        associative_value_tokens=2,
        associative_top_k=2,
        associative_retrieval_threshold=0.5,
        associative_write_threshold=0.0,
        associative_protected_fraction=0.125,
        associative_candidate_fraction=0.25,
    )


def test_memory_capacity_is_bounded() -> None:
    torch.manual_seed(20)
    model = SynchronousControlProcessor(memory_config()).eval()
    state = None
    for _ in range(24):
        _, state = model.forward_chunk(torch.rand(1, 1, 3, 16, 16), state=state)

    assert state.associative is not None
    assert state.associative.keys.shape[1] == 8
    assert int(state.associative.occupied.sum()) <= 8


def test_similar_stimulus_retrieves_an_association() -> None:
    torch.manual_seed(21)
    model = SynchronousControlProcessor(memory_config()).eval()
    stimulus = torch.rand(1, 1, 3, 16, 16)
    _, remembered = model.forward_chunk(stimulus)
    assert remembered.associative is not None
    assert remembered.associative.occupied.any()

    empty = model.init_state(1, stimulus.device, stimulus.dtype)
    empty.layers = remembered.layers
    recalled_output, _ = model.forward_chunk(
        stimulus, state=remembered, update_associative_memory=False
    )
    empty_output, _ = model.forward_chunk(
        stimulus, state=empty, update_associative_memory=False
    )
    assert not torch.allclose(recalled_output, empty_output)


def test_associative_memory_can_be_saved_and_restored(tmp_path) -> None:
    torch.manual_seed(22)
    model = SynchronousControlProcessor(memory_config()).eval()
    _, state = model.forward_chunk(torch.rand(1, 2, 3, 16, 16))
    path = tmp_path / "long_term_memory.pt"

    model.save_associative_memory(str(path), state)
    restored = model.load_associative_memory(str(path), torch.device("cpu"))

    assert state.associative is not None
    torch.testing.assert_close(restored.keys, state.associative.keys)
    torch.testing.assert_close(restored.values, state.associative.values)
    assert torch.equal(restored.occupied, state.associative.occupied)


def test_disabling_writes_keeps_memory_immutable() -> None:
    model = SynchronousControlProcessor(memory_config()).eval()
    state = model.init_state(1, torch.device("cpu"))
    _, next_state = model.forward_chunk(
        torch.rand(1, 3, 3, 16, 16),
        state=state,
        update_associative_memory=False,
    )
    assert next_state.associative is not None
    assert not next_state.associative.occupied.any()
