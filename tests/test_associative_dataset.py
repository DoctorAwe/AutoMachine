from __future__ import annotations

import torch

from automachine.synthetic import EpisodicAssociationDataset


def test_episode_shapes_and_gap() -> None:
    data = EpisodicAssociationDataset(
        samples=2, image_size=16, cue_frames=2, gap=20, query_frames=2
    )
    video, auxiliary, target, salience = data[0]
    assert video.shape == (30, 3, 16, 16)
    assert auxiliary.shape == (30, 4)
    assert target.shape == ()
    assert salience.shape == (30,)
    assert torch.all(salience[:8] == 1)
    assert torch.all(salience[8:] == 0)
    assert torch.all(auxiliary[8:] == 0)


def test_episode_mapping_changes_across_samples() -> None:
    data = EpisodicAssociationDataset(samples=16, image_size=16, gap=20)
    learning_labels = []
    for index in range(16):
        _, auxiliary, _, _ = data[index]
        learning_labels.append(tuple(auxiliary[::2][:4].argmax(-1).tolist()))
    assert len(set(learning_labels)) > 1
