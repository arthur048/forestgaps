"""
Module de modèles pour la détection des trouées forestières.

Ce module fournit différentes architectures de réseaux de neurones
pour la segmentation d'images de télédétection forestière.
"""

# Importer les architectures U-Net
from models.unet.basic import UNet
from models.unet.residual import ResUNet
from models.unet.attention import AttentionUNet
from models.unet.film import FiLMUNet
from models.unet.advanced import UNet3Plus

# Importer les blocs de base
from models.blocks.convolution import ConvBlock, DoubleConvBlock
from models.blocks.pooling import DownsampleBlock, UpsampleBlock
from models.blocks.residual import ResidualBlock, BottleneckBlock

# Importer les mécanismes d'attention
from models.attention.cbam import CBAM, ChannelAttention, SpatialAttention
from models.attention.self_attention import SelfAttention

# Importer les couches FiLM
from models.film.layers import FiLMLayer, FiLMGenerator, AdaptiveFiLM
from models.film.blocks import FiLMBlock, FiLMResidualBlock, ConditionedBlock

__all__ = [
    # Architectures U-Net
    'UNet',
    'ResUNet',
    'AttentionUNet',
    'FiLMUNet',
    'UNet3Plus',
    
    # Blocs de base
    'ConvBlock',
    'DoubleConvBlock',
    'DownsampleBlock',
    'UpsampleBlock',
    'ResidualBlock',
    'BottleneckBlock',
    
    # Mécanismes d'attention
    'CBAM',
    'ChannelAttention',
    'SpatialAttention',
    'SelfAttention',
    
    # Couches FiLM
    'FiLMLayer',
    'FiLMGenerator',
    'AdaptiveFiLM',
    'FiLMBlock',
    'FiLMResidualBlock',
    'ConditionedBlock'
]
