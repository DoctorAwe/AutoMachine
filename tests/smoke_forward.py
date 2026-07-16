from __future__ import annotations

import torch

from automachine import SynchronousControlConfig, SynchronousControlProcessor


def main() -> None:
    config = SynchronousControlConfig(
        image_size=96,
        response_dim=16,
        frames_per_step=2,
        token_dim=64,
        state_tokens=(16, 24, 32),
    )
    model = SynchronousControlProcessor(config)
    video = torch.rand(2, 4, 3, 96, 96)
    response, state = model.forward_chunk(video)
    assert response.shape == (2, 2, 16)
    assert state.step == 2
    print(f"OK response={tuple(response.shape)} memory_step={state.step}")


if __name__ == "__main__":
    main()
