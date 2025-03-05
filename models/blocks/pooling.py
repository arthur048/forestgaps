"""
Blocs de pooling et upsampling pour les architectures U-Net.

Ce module fournit des implémentations des blocs de réduction et d'augmentation
de résolution qui sont utilisés dans les architectures U-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.convolution import ConvBlock


class DownsampleBlock(nn.Module):
    """
    Bloc de réduction de résolution (downsampling) utilisé dans l'encodeur U-Net.
    
    Ce bloc peut utiliser soit un MaxPool2d, soit une convolution avec stride 2
    pour réduire la résolution spatiale.
    
    Attributes:
        downsample: Couche de réduction de résolution (MaxPool2d ou Conv2d)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        pool_type: str = 'max',
        pool_size: int = 2,
        stride: int = 2,
        with_conv: bool = False,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU
    ):
        """
        Initialise un bloc de downsampling.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie (utilisé seulement si with_conv=True)
            pool_type: Type de pooling ('max', 'avg' ou 'conv')
            pool_size: Taille du noyau de pooling
            stride: Pas du pooling
            with_conv: Si True, ajoute une convolution après le pooling
            norm_layer: Couche de normalisation pour la convolution (si with_conv=True)
            activation: Fonction d'activation pour la convolution (si with_conv=True)
        """
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
            
        if pool_type == 'max':
            self.downsample = nn.MaxPool2d(kernel_size=pool_size, stride=stride)
        elif pool_type == 'avg':
            self.downsample = nn.AvgPool2d(kernel_size=pool_size, stride=stride)
        elif pool_type == 'conv':
            self.downsample = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=pool_size,
                stride=stride,
                padding=0
            )
        else:
            raise ValueError(f"Type de pooling inconnu: {pool_type}")
            
        if with_conv and pool_type != 'conv':
            self.conv = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=norm_layer,
                activation=activation
            )
        else:
            self.conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc de downsampling.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur de sortie avec résolution réduite
        """
        x = self.downsample(x)
        
        if self.conv:
            x = self.conv(x)
            
        return x


class UpsampleBlock(nn.Module):
    """
    Bloc d'augmentation de résolution (upsampling) utilisé dans le décodeur U-Net.
    
    Ce bloc peut utiliser soit une interpolation, soit une ConvTranspose2d
    pour augmenter la résolution spatiale.
    
    Attributes:
        upsample: Couche d'augmentation de résolution
        conv: Couche de convolution optionnelle
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        mode: str = 'bilinear',
        align_corners: bool = True,
        with_conv: bool = False,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU
    ):
        """
        Initialise un bloc d'upsampling.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            scale_factor: Facteur d'agrandissement
            mode: Mode d'interpolation ('nearest', 'bilinear', 'transpose')
            align_corners: Si True, aligne les coins pour l'interpolation
            with_conv: Si True, ajoute une convolution après l'upsampling
            norm_layer: Couche de normalisation pour la convolution (si with_conv=True)
            activation: Fonction d'activation pour la convolution (si with_conv=True)
        """
        super().__init__()
        
        if mode == 'transpose':
            self.upsample = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=scale_factor,
                stride=scale_factor,
                padding=0
            )
        else:
            self.upsample = nn.Upsample(
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners if mode in ['bilinear', 'bicubic'] else None
            )
            
        if with_conv and mode != 'transpose':
            self.conv = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=norm_layer,
                activation=activation
            )
        else:
            self.conv = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc d'upsampling.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur de sortie avec résolution augmentée
        """
        x = self.upsample(x)
        
        if self.conv:
            x = self.conv(x)
            
        return x 