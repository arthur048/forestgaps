"""Configuration loader with YAML support and Pydantic validation.

Conforme "Audit du workflow PyTorch": externalized configuration system
with validation, inheritance, and type checking.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .schemas import TrainingConfig, DataConfig, ModelConfig


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary with configuration

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is not valid YAML
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def save_yaml(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Path to save YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_training_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> TrainingConfig:
    """Load and validate training configuration.

    Args:
        config_path: Path to training.yaml (None = use default)
        overrides: Dictionary of values to override

    Returns:
        Validated TrainingConfig instance

    Example:
        >>> config = load_training_config("configs/my_training.yaml")
        >>> config = load_training_config(overrides={"epochs": 100, "batch_size": 32})
    """
    # Load default config
    default_path = Path(__file__).parent.parent.parent / "configs" / "defaults" / "training.yaml"
    config = load_yaml(default_path)

    # Load custom config if provided
    if config_path is not None:
        custom_config = load_yaml(config_path)
        config = merge_configs(config, custom_config)

    # Apply overrides
    if overrides is not None:
        config = merge_configs(config, overrides)

    # Validate with Pydantic
    return TrainingConfig(**config)


def load_data_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> DataConfig:
    """Load and validate data configuration.

    Args:
        config_path: Path to data.yaml (None = use default)
        overrides: Dictionary of values to override

    Returns:
        Validated DataConfig instance

    Example:
        >>> config = load_data_config("configs/my_data.yaml")
        >>> config = load_data_config(overrides={"preprocessing": {"tile_size": 512}})
    """
    # Load default config
    default_path = Path(__file__).parent.parent.parent / "configs" / "defaults" / "data.yaml"
    config = load_yaml(default_path)

    # Load custom config if provided
    if config_path is not None:
        custom_config = load_yaml(config_path)
        config = merge_configs(config, custom_config)

    # Apply overrides
    if overrides is not None:
        config = merge_configs(config, overrides)

    # Validate with Pydantic
    return DataConfig(**config)


def load_model_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> ModelConfig:
    """Load and validate model configuration.

    Args:
        config_path: Path to model.yaml (None = use default)
        overrides: Dictionary of values to override

    Returns:
        Validated ModelConfig instance

    Example:
        >>> config = load_model_config("configs/my_model.yaml")
        >>> config = load_model_config(overrides={"model_type": "deeplabv3_plus"})
    """
    # Load default config
    default_path = Path(__file__).parent.parent.parent / "configs" / "defaults" / "model.yaml"
    config = load_yaml(default_path)

    # Load custom config if provided
    if config_path is not None:
        custom_config = load_yaml(config_path)
        config = merge_configs(config, custom_config)

    # Apply overrides
    if overrides is not None:
        config = merge_configs(config, overrides)

    # Validate with Pydantic
    return ModelConfig(**config)


def load_complete_config(
    training_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
    model_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[TrainingConfig, DataConfig, ModelConfig]]:
    """Load complete configuration (training + data + model).

    Args:
        training_path: Path to training.yaml
        data_path: Path to data.yaml
        model_path: Path to model.yaml
        overrides: Dictionary with overrides for each config type

    Returns:
        Dictionary with 'training', 'data', and 'model' configs

    Example:
        >>> config = load_complete_config(
        ...     training_path="configs/training.yaml",
        ...     overrides={"training": {"epochs": 100}, "model": {"model_type": "unet"}}
        ... )
        >>> training_config = config["training"]
        >>> data_config = config["data"]
        >>> model_config = config["model"]
    """
    overrides = overrides or {}

    return {
        "training": load_training_config(
            training_path,
            overrides.get("training")
        ),
        "data": load_data_config(
            data_path,
            overrides.get("data")
        ),
        "model": load_model_config(
            model_path,
            overrides.get("model")
        ),
    }


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Configuration to override with

    Returns:
        Merged configuration

    Example:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> override = {"b": {"c": 99}, "e": 4}
        >>> merge_configs(base, override)
        {'a': 1, 'b': {'c': 99, 'd': 3}, 'e': 4}
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def config_to_dict(config: Union[TrainingConfig, DataConfig, ModelConfig]) -> Dict[str, Any]:
    """Convert Pydantic config to dictionary.

    Args:
        config: Pydantic config instance

    Returns:
        Configuration as dictionary
    """
    return config.dict()


def save_config(
    config: Union[TrainingConfig, DataConfig, ModelConfig],
    path: Union[str, Path]
) -> None:
    """Save Pydantic config to YAML file.

    Args:
        config: Pydantic config instance
        path: Path to save YAML file
    """
    config_dict = config_to_dict(config)
    save_yaml(config_dict, path)
