"""Data configuration schemas.

Conforme Document 2 "Audit du workflow PyTorch": data pipeline configuration
with preprocessing, augmentation, and normalization.
"""

from typing import Optional, List, Literal, Tuple
from pydantic import BaseModel, Field, validator


class PreprocessingConfig(BaseModel):
    """Raster preprocessing configuration."""

    # Tile generation
    tile_size: int = Field(256, description="Tile size in pixels", ge=32)
    overlap: float = Field(0.0, description="Tile overlap fraction", ge=0.0, lt=1.0)

    # Normalization
    normalize_method: Literal["minmax", "standard", "per_tile"] = Field(
        "per_tile",
        description="Normalization method"
    )
    normalize_range: Tuple[float, float] = Field(
        (0.0, 1.0),
        description="Target range for normalization"
    )

    # Filtering
    min_valid_pixels: Optional[int] = Field(
        None,
        description="Minimum valid (non-NaN) pixels per tile (None = no filtering)"
    )
    min_gap_pixels: Optional[int] = Field(
        None,
        description="Minimum gap pixels per tile for segmentation (None = no filtering)"
    )

    class Config:
        extra = "forbid"


class AugmentationConfig(BaseModel):
    """Data augmentation configuration."""

    enabled: bool = Field(True, description="Enable data augmentation")

    # Geometric transformations
    random_flip_prob: float = Field(0.5, description="Random flip probability", ge=0.0, le=1.0)
    random_rotation: bool = Field(True, description="Enable random rotation")
    random_rotation_degrees: float = Field(15.0, description="Max rotation degrees", ge=0.0)

    # Intensity transformations
    brightness_jitter: Optional[float] = Field(
        0.2,
        description="Brightness jitter factor (None = disabled)",
        ge=0.0
    )
    contrast_jitter: Optional[float] = Field(
        0.2,
        description="Contrast jitter factor (None = disabled)",
        ge=0.0
    )

    # Gaussian noise
    gaussian_noise_prob: float = Field(0.2, description="Gaussian noise probability", ge=0.0, le=1.0)
    gaussian_noise_std: float = Field(0.01, description="Gaussian noise std", ge=0.0)

    # Gaussian blur
    gaussian_blur_prob: float = Field(0.2, description="Gaussian blur probability", ge=0.0, le=1.0)
    gaussian_blur_sigma: Tuple[float, float] = Field(
        (0.1, 2.0),
        description="Gaussian blur sigma range"
    )

    # Advanced augmentations (Kornia)
    use_kornia: bool = Field(False, description="Use Kornia for GPU augmentations")

    class Config:
        extra = "forbid"


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    # Data splits
    train_split: float = Field(0.7, description="Training split fraction", gt=0.0, lt=1.0)
    val_split: float = Field(0.15, description="Validation split fraction", gt=0.0, lt=1.0)
    test_split: float = Field(0.15, description="Test split fraction", gt=0.0, lt=1.0)

    # Gap detection thresholds
    thresholds: List[float] = Field(
        [2.0, 5.0, 10.0],
        description="Gap height thresholds (meters)"
    )

    # Dataset behavior
    cache_in_memory: bool = Field(False, description="Cache dataset in memory")
    shuffle_train: bool = Field(True, description="Shuffle training data")
    shuffle_val: bool = Field(False, description="Shuffle validation data")

    @validator("train_split", "val_split", "test_split")
    def validate_splits_sum(cls, v, values):
        """Ensure splits sum to approximately 1."""
        if "train_split" in values and "val_split" in values and "test_split" in values:
            total = values["train_split"] + values["val_split"] + values["test_split"]
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"Data splits must sum to 1, got {total}")
        return v

    class Config:
        extra = "forbid"


class DataConfig(BaseModel):
    """Complete data configuration.

    Conforme Document 3 "Matériel et Méthode": data pipeline with preprocessing,
    augmentation, and dataset splits.
    """

    # Data paths
    data_dir: str = Field("data", description="Root data directory")
    processed_dir: str = Field("data/processed", description="Processed data directory")
    tiles_dir: str = Field("data/tiles", description="Tiles directory")

    # Input channels
    in_channels: int = Field(1, description="Number of input channels", ge=1)
    input_type: Literal["dsm", "chm", "both"] = Field(
        "dsm",
        description="Input raster type"
    )

    # Sub-configurations
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

    # DataLoader parameters
    num_workers: int = Field(4, description="Number of DataLoader workers", ge=0)
    pin_memory: bool = Field(True, description="Pin memory for faster GPU transfer")
    prefetch_factor: int = Field(2, description="DataLoader prefetch factor", ge=1)
    persistent_workers: bool = Field(True, description="Keep workers alive between epochs")

    class Config:
        extra = "allow"
