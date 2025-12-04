"""Configuration schemas with Pydantic validation.

Conforme "Audit du workflow PyTorch": complete configuration system with
validation, type checking, and YAML support.
"""

from .training_schema import (
    TrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
    LossConfig,
    CallbackConfig,
    OptimizationConfig,
)
from .data_schema import (
    DataConfig,
    PreprocessingConfig,
    AugmentationConfig,
    DatasetConfig,
)
from .model_schema import ModelConfig

__all__ = [
    # Training schemas
    "TrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "LossConfig",
    "CallbackConfig",
    "OptimizationConfig",
    # Data schemas
    "DataConfig",
    "PreprocessingConfig",
    "AugmentationConfig",
    "DatasetConfig",
    # Model schemas
    "ModelConfig",
]
