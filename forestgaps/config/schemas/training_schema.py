"""Training configuration schemas.

Conforme "Audit du workflow PyTorch": complete training configuration with
optimizer, scheduler, callbacks, losses, and optimizations.
"""

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, validator


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""

    type: Literal["adam", "adamw", "sgd", "rmsprop"] = Field(
        "adamw",
        description="Optimizer type"
    )
    lr: float = Field(0.001, description="Learning rate", gt=0.0)
    weight_decay: float = Field(0.01, description="Weight decay (L2 penalty)", ge=0.0)

    # Adam/AdamW specific
    betas: tuple = Field((0.9, 0.999), description="Adam beta parameters")
    eps: float = Field(1e-8, description="Adam epsilon", gt=0.0)

    # SGD specific
    momentum: float = Field(0.9, description="SGD momentum", ge=0.0, le=1.0)
    nesterov: bool = Field(True, description="Use Nesterov momentum")

    class Config:
        extra = "forbid"


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    type: Literal["onecycle", "cosine", "reduce_on_plateau", "step", "exponential", "none"] = Field(
        "onecycle",
        description="Scheduler type"
    )

    # OneCycleLR
    max_lr: Optional[float] = Field(0.01, description="OneCycleLR max learning rate")
    pct_start: float = Field(0.3, description="OneCycleLR percentage of cycle spent increasing lr")
    div_factor: float = Field(25.0, description="OneCycleLR initial_lr = max_lr/div_factor")
    final_div_factor: float = Field(1e4, description="OneCycleLR min_lr = initial_lr/final_div_factor")

    # CosineAnnealingLR
    T_max: Optional[int] = Field(None, description="CosineAnnealing period")
    eta_min: float = Field(0.0, description="CosineAnnealing minimum lr")

    # ReduceLROnPlateau
    mode: Literal["min", "max"] = Field("min", description="ReduceLROnPlateau mode")
    factor: float = Field(0.1, description="ReduceLROnPlateau factor", gt=0.0, lt=1.0)
    patience: int = Field(10, description="ReduceLROnPlateau patience", ge=0)
    threshold: float = Field(1e-4, description="ReduceLROnPlateau threshold")
    monitor: str = Field("val_loss", description="Metric to monitor")

    # StepLR
    step_size: int = Field(30, description="StepLR step size", ge=1)
    gamma: float = Field(0.1, description="StepLR/ExponentialLR gamma", gt=0.0, le=1.0)

    # Warmup
    warmup_epochs: int = Field(0, description="Number of warmup epochs", ge=0)
    warmup_start_lr: float = Field(1e-6, description="Starting lr for warmup", gt=0.0)

    class Config:
        extra = "forbid"


class LossConfig(BaseModel):
    """Loss function configuration."""

    type: Literal["combo", "bce", "dice", "focal"] = Field(
        "combo",
        description="Loss function type"
    )

    # ComboLoss weights (must sum to 1)
    bce_weight: float = Field(0.5, description="BCE weight in ComboLoss", ge=0.0, le=1.0)
    dice_weight: float = Field(0.3, description="Dice weight in ComboLoss", ge=0.0, le=1.0)
    focal_weight: float = Field(0.2, description="Focal weight in ComboLoss", ge=0.0, le=1.0)

    # FocalLoss parameters
    focal_alpha: float = Field(0.25, description="Focal loss alpha", ge=0.0, le=1.0)
    focal_gamma: float = Field(2.0, description="Focal loss gamma", ge=0.0)

    # DiceLoss parameters
    smooth: float = Field(1.0, description="Dice loss smoothing factor", gt=0.0)

    @validator("bce_weight", "dice_weight", "focal_weight", always=True)
    def validate_weights_sum(cls, v, values):
        """Ensure loss weights sum to 1 for ComboLoss."""
        if "type" in values and values["type"] == "combo":
            total = values.get("bce_weight", 0.5) + values.get("dice_weight", 0.3) + values.get("focal_weight", 0.2)
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                raise ValueError(f"ComboLoss weights must sum to 1, got {total}")
        return v

    class Config:
        extra = "forbid"


class CallbackConfig(BaseModel):
    """Training callbacks configuration."""

    # Early Stopping
    early_stopping: bool = Field(True, description="Enable early stopping")
    early_stopping_monitor: str = Field("val_loss", description="Metric to monitor")
    early_stopping_patience: int = Field(10, description="Early stopping patience", ge=1)
    early_stopping_mode: Literal["min", "max"] = Field("min", description="Early stopping mode")
    early_stopping_min_delta: float = Field(0.0, description="Minimum change to qualify as improvement")

    # Model Checkpoint
    checkpoint_save_best_only: bool = Field(True, description="Save only best model")
    checkpoint_monitor: str = Field("val_loss", description="Metric to monitor for best model")
    checkpoint_mode: Literal["min", "max"] = Field("min", description="Checkpoint mode")
    checkpoint_save_frequency: int = Field(1, description="Save checkpoint every N epochs", ge=1)

    # TensorBoard
    tensorboard_enabled: bool = Field(True, description="Enable TensorBoard logging")
    tensorboard_log_dir: str = Field("runs", description="TensorBoard log directory")
    tensorboard_comment: str = Field("", description="TensorBoard run comment")

    # Progress Bar
    progress_bar: bool = Field(True, description="Enable progress bar")

    class Config:
        extra = "forbid"


class OptimizationConfig(BaseModel):
    """Training optimizations configuration."""

    # Automatic Mixed Precision
    use_amp: bool = Field(True, description="Use Automatic Mixed Precision (AMP)")

    # Gradient Clipping
    gradient_clip_value: Optional[float] = Field(
        None,
        description="Gradient clipping value (None = disabled)",
        gt=0.0
    )
    gradient_clip_norm: Optional[float] = Field(
        1.0,
        description="Gradient norm clipping (None = disabled)",
        gt=0.0
    )

    # Gradient Accumulation
    accumulate_grad_batches: int = Field(
        1,
        description="Accumulate gradients over N batches",
        ge=1
    )

    # Gradient Checkpointing
    use_gradient_checkpointing: bool = Field(
        False,
        description="Use gradient checkpointing to save memory"
    )

    # torch.compile()
    use_torch_compile: bool = Field(
        False,
        description="Use torch.compile() for optimization (PyTorch 2.0+)"
    )
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        "default",
        description="torch.compile() mode"
    )

    class Config:
        extra = "forbid"


class TrainingConfig(BaseModel):
    """Complete training configuration.

    Conforme Document 2 "Audit du workflow PyTorch": comprehensive training
    configuration with optimizer, scheduler, callbacks, and optimizations.
    """

    # Basic training parameters
    epochs: int = Field(50, description="Maximum number of epochs", ge=1)
    batch_size: int = Field(16, description="Training batch size", ge=1)
    val_batch_size: Optional[int] = Field(None, description="Validation batch size (None = same as batch_size)")

    # Device and workers
    device: Literal["cuda", "cpu", "auto"] = Field("auto", description="Device to use")
    num_workers: int = Field(4, description="DataLoader workers", ge=0)
    pin_memory: bool = Field(True, description="Pin memory for faster GPU transfer")

    # Seed for reproducibility
    seed: Optional[int] = Field(42, description="Random seed for reproducibility")

    # Sub-configurations
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    callbacks: CallbackConfig = Field(default_factory=CallbackConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)

    # Directories
    save_dir: str = Field("outputs", description="Directory to save outputs")
    checkpoint_dir: str = Field("checkpoints", description="Directory to save checkpoints")
    log_dir: str = Field("logs", description="Directory to save logs")

    class Config:
        extra = "allow"  # Allow additional fields for extensibility

    @validator("val_batch_size", always=True)
    def set_val_batch_size(cls, v, values):
        """Set val_batch_size to batch_size if not specified."""
        return v if v is not None else values.get("batch_size", 16)
