"""Model configuration schemas.

Conforme Document 1 & 2: model architecture configuration for all ForestGaps models.
"""

from typing import Optional, Literal, List
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Model architecture configuration.

    Supports all ForestGaps models: UNet, UNetFiLM, DeepLabV3+, AttentionUNet,
    ResUNet, etc.
    """

    # Model type
    model_type: Literal[
        "unet",
        "unet_film",
        "film_unet",  # Alias for unet_film
        "deeplabv3_plus",
        "attention_unet",
        "res_unet",
        "res_unet_film",
        "regression_unet"
    ] = Field("unet_film", description="Model architecture type")

    # Task type
    task: Literal["segmentation", "regression"] = Field(
        "segmentation",
        description="Task type"
    )

    # Basic architecture parameters
    in_channels: int = Field(1, description="Number of input channels", ge=1)
    out_channels: int = Field(1, description="Number of output channels", ge=1)
    base_channels: int = Field(64, description="Base number of channels", ge=16)
    depth: int = Field(4, description="Network depth (encoder/decoder stages)", ge=1, le=5)

    # FiLM conditioning (for unet_film, res_unet_film)
    num_conditions: int = Field(1, description="Number of FiLM conditions", ge=1)

    # Attention mechanisms
    use_attention: bool = Field(False, description="Use attention gates (for attention_unet)")
    attention_type: Optional[Literal["cbam", "se", "attention_gate"]] = Field(
        None,
        description="Attention mechanism type"
    )

    # Regularization
    dropout_rate: float = Field(0.1, description="Dropout rate", ge=0.0, le=0.5)
    droppath_rate: float = Field(0.0, description="DropPath rate", ge=0.0, le=0.5)

    # BatchNorm/GroupNorm
    norm_type: Literal["batch", "group", "instance"] = Field(
        "batch",
        description="Normalization type"
    )
    num_groups: int = Field(8, description="Number of groups for GroupNorm", ge=1)

    # Activation function
    activation: Literal["relu", "leaky_relu", "gelu", "silu"] = Field(
        "relu",
        description="Activation function"
    )

    # DeepLabV3+ specific
    aspp_dilations: List[int] = Field(
        [6, 12, 18],
        description="ASPP dilation rates (for deeplabv3_plus)"
    )
    low_level_channels: int = Field(
        48,
        description="Low-level feature channels (for deeplabv3_plus)"
    )

    # Output activation
    output_activation: Optional[Literal["sigmoid", "softmax", "none"]] = Field(
        "sigmoid",
        description="Output activation function"
    )

    # Model initialization
    init_method: Literal["kaiming", "xavier", "orthogonal", "default"] = Field(
        "kaiming",
        description="Weight initialization method"
    )

    # Pretrained weights
    pretrained: bool = Field(False, description="Use pretrained weights")
    pretrained_path: Optional[str] = Field(
        None,
        description="Path to pretrained weights"
    )

    @validator("task", always=True)
    def validate_task_model(cls, v, values):
        """Ensure task matches model type."""
        model_type = values.get("model_type")
        if model_type == "regression_unet" and v != "regression":
            raise ValueError("regression_unet requires task='regression'")
        if model_type != "regression_unet" and v == "regression":
            raise ValueError("Only regression_unet supports task='regression'")
        return v

    @validator("out_channels", always=True)
    def validate_out_channels_task(cls, v, values):
        """Ensure out_channels is 1 for binary segmentation."""
        task = values.get("task")
        if task == "segmentation" and v != 1:
            raise ValueError("Segmentation task requires out_channels=1 (binary)")
        return v

    class Config:
        extra = "forbid"
