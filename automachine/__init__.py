"""AutoMachine neural stream processor prototype."""

from .model import NeuralStreamProcessor, NeuralStreamProcessorConfig
from .synthetic import MovingBlobVideoDataset

__all__ = [
    "MovingBlobVideoDataset",
    "NeuralStreamProcessor",
    "NeuralStreamProcessorConfig",
]
