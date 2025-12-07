"""
Custom exceptions for the MistakeDetection project.

Questo modulo fornisce una gerarchia di eccezioni personalizzate 
organizzate per dominio (dataset, model, training, etc.)
"""

from .base import MistakeDetectionError
from .dataset_exceptions import (
    DatasetError,
    AnnotationNotFoundError,
    FeatureFileNotFoundError,
    InvalidDatasetSourceError,
    MismatchedDataShapeError,
    EmptyDatasetError,
    CorruptedFeatureFileError,
    MissingAnnotationKeyError
)
from .model_exceptions import (
    ModelError,
    InvalidModelConfigError,
    ModelLoadError,
    ModelSaveError,
    InvalidInputShapeError,
)
from .training_exceptions import (
    TrainingError,
    CheckpointError,
    ValidationError,
    MetricCalculationError,
)
from .config_exceptions import (
    ConfigError,
    InvalidConfigValueError,
    MissingConfigKeyError,
)

__all__ = [
    # Base
    "MistakeDetectionError",
    # Dataset
    "DatasetError",
    "AnnotationNotFoundError",
    "FeatureFileNotFoundError",
    "InvalidDatasetSourceError",
    "MismatchedDataShapeError",
    "EmptyDatasetError",
    "CorruptedFeatureFileError",
    "MissingAnnotationKeyError",
    # Model
    "ModelError",
    "InvalidModelConfigError",
    "ModelLoadError",
    "ModelSaveError",
    "InvalidInputShapeError",
    # Training
    "TrainingError",
    "CheckpointError",
    "ValidationError",
    "MetricCalculationError",
    # Config
    "ConfigError",
    "InvalidConfigValueError",
    "MissingConfigKeyError",
]
