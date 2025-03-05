"""
Blocs réutilisables pour la construction d'architectures de réseaux neuronaux.

Ce module fournit des blocs de construction modulaires qui peuvent être
utilisés pour assembler différentes architectures de modèles de segmentation.
"""

from .conv import (
    ConvBlock,
    DoubleConvBlock,
    ResidualBlock,
    SEBlock
)

from .downsampling import (
    DownsampleBlock,
    MaxPoolDownsample,
    StridedConvDownsample
)

from .upsampling import (
    UpsampleBlock,
    TransposeConvUpsampling,
    BilinearUpsampling,
    PixelShuffleUpsampling
)

from .attention import (
    AttentionGate,
    SpatialAttentionBlock,
    ChannelAttentionBlock,
    CBAM
)

from .droppath import (
    DropPath,
    DropPathScheduler,
    train_with_droppath_scheduling
)

from .residual_advanced import (
    ResidualBlockWithDropPath,
    ResidualBlockWithCBAM,
    FiLMResidualBlock,
    ResidualBlockWithFiLMCBAMDropPath
)

__all__ = [
    # Blocs de convolution
    "ConvBlock",
    "DoubleConvBlock",
    "ResidualBlock",
    "SEBlock",
    
    # Blocs de sous-échantillonnage
    "DownsampleBlock",
    "MaxPoolDownsample",
    "StridedConvDownsample",
    
    # Blocs de sur-échantillonnage
    "UpsampleBlock",
    "TransposeConvUpsampling",
    "BilinearUpsampling",
    "PixelShuffleUpsampling",
    
    # Mécanismes d'attention
    "AttentionGate",
    "SpatialAttentionBlock",
    "ChannelAttentionBlock",
    "CBAM",
    
    # Mécanismes de DropPath
    "DropPath",
    "DropPathScheduler",
    "train_with_droppath_scheduling",
    
    # Blocs résiduels avancés
    "ResidualBlockWithDropPath",
    "ResidualBlockWithCBAM",
    "FiLMResidualBlock",
    "ResidualBlockWithFiLMCBAMDropPath"
] 