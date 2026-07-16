"""AutoMachine neural stream processor prototype."""

from .model import NeuralStreamProcessor, NeuralStreamProcessorConfig
from .feature_model import FeatureStreamProcessor, FeatureStreamProcessorConfig
from .synthetic import MovingBlobVideoDataset

__all__ = [
    "FeatureStreamProcessor",
    "FeatureStreamProcessorConfig",
    "MovingBlobVideoDataset",
    "NeuralStreamProcessor",
    "NeuralStreamProcessorConfig",
]
