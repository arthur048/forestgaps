"""
Blocs de sous-échantillonnage pour les architectures de réseaux neuronaux.

Ce module fournit différentes implémentations de blocs de sous-échantillonnage
(downsampling) qui peuvent être utilisés dans les architectures d'encodeur-décodeur
comme U-Net pour réduire la résolution spatiale des caractéristiques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Optional, Tuple, Union

from .conv import ConvBlock


class DownsampleBlock(nn.Module):
    """
    Interface de base pour les blocs de sous-échantillonnage.
    
    Cette classe définit l'interface commune pour tous les blocs
    de sous-échantillonnage et peut être étendue pour créer différentes
    implémentations.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2
    ):
        """
        Initialise un bloc de sous-échantillonnage.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            scale_factor: Facteur d'échelle pour le sous-échantillonnage
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc de sous-échantillonnage.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur sous-échantillonné [B, C', H/scale, W/scale]
        """
        raise NotImplementedError("Les sous-classes doivent implémenter cette méthode")


class MaxPoolDownsample(DownsampleBlock):
    """
    Bloc de sous-échantillonnage utilisant MaxPool2d.
    
    Ce bloc effectue un sous-échantillonnage en utilisant une opération
    de pooling maximum, suivie éventuellement d'une convolution pour
    ajuster le nombre de canaux.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        kernel_size: int = 2,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU
    ):
        """
        Initialise un bloc de sous-échantillonnage MaxPool.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            scale_factor: Facteur d'échelle pour le sous-échantillonnage
            kernel_size: Taille du noyau de pooling
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
        """
        super().__init__(in_channels, out_channels, scale_factor)
        
        # Pooling pour le sous-échantillonnage
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=scale_factor)
        
        # Convolution pour ajuster le nombre de canaux si nécessaire
        self.conv = None
        if in_channels != out_channels:
            self.conv = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,  # Convolution 1x1
                stride=1,
                norm_layer=norm_layer,
                activation=activation
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc MaxPool.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur sous-échantillonné [B, C', H/scale, W/scale]
        """
        x = self.pool(x)
        
        if self.conv is not None:
            x = self.conv(x)
            
        return x


class StridedConvDownsample(DownsampleBlock):
    """
    Bloc de sous-échantillonnage utilisant une convolution à pas (stride).
    
    Ce bloc effectue un sous-échantillonnage en utilisant une convolution
    avec un stride > 1, ce qui permet d'apprendre les paramètres de 
    sous-échantillonnage.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        kernel_size: int = 3,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU
    ):
        """
        Initialise un bloc de sous-échantillonnage par convolution.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            scale_factor: Facteur d'échelle pour le sous-échantillonnage (stride)
            kernel_size: Taille du noyau de convolution
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
        """
        super().__init__(in_channels, out_channels, scale_factor)
        
        # Le padding doit être calculé pour maintenir des dimensions correctes
        padding = kernel_size // 2
        
        # Convolution avec stride pour le sous-échantillonnage
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=scale_factor,
            padding=padding,
            norm_layer=norm_layer,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc de convolution à pas.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur sous-échantillonné [B, C', H/scale, W/scale]
        """
        return self.conv(x)


class AvgPoolDownsample(DownsampleBlock):
    """
    Bloc de sous-échantillonnage utilisant AvgPool2d.
    
    Ce bloc effectue un sous-échantillonnage en utilisant une opération
    de pooling moyen, suivie éventuellement d'une convolution pour
    ajuster le nombre de canaux.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        kernel_size: int = 2,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU
    ):
        """
        Initialise un bloc de sous-échantillonnage AvgPool.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            scale_factor: Facteur d'échelle pour le sous-échantillonnage
            kernel_size: Taille du noyau de pooling
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
        """
        super().__init__(in_channels, out_channels, scale_factor)
        
        # Pooling pour le sous-échantillonnage
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=scale_factor)
        
        # Convolution pour ajuster le nombre de canaux si nécessaire
        self.conv = None
        if in_channels != out_channels:
            self.conv = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,  # Convolution 1x1
                stride=1,
                norm_layer=norm_layer,
                activation=activation
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc AvgPool.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur sous-échantillonné [B, C', H/scale, W/scale]
        """
        x = self.pool(x)
        
        if self.conv is not None:
            x = self.conv(x)
            
        return x 