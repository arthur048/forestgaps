"""
Blocs de convolution pour les architectures U-Net.

Ce module fournit des implémentations de blocs de convolution
qui sont utilisés comme composants de base dans les architectures U-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Bloc de convolution simple avec normalisation et activation.
    
    Attributes:
        conv: Couche de convolution 2D
        norm: Couche de normalisation (BatchNorm2d par défaut)
        activation: Fonction d'activation (ReLU par défaut)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU,
        dropout_rate: float = 0.0
    ):
        """
        Initialise un bloc de convolution.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            kernel_size: Taille du noyau de convolution
            stride: Pas de la convolution
            padding: Remplissage appliqué aux bords de l'image
            bias: Si True, ajoute un terme de biais
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
            dropout_rate: Taux de dropout (0 pour désactiver)
        """
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
        self.norm = norm_layer(out_channels) if norm_layer else None
        self.activation = activation(inplace=True) if activation else None
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc de convolution.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur de sortie après convolution, normalisation et activation
        """
        x = self.conv(x)
        
        if self.norm:
            x = self.norm(x)
        
        if self.activation:
            x = self.activation(x)
            
        if self.dropout:
            x = self.dropout(x)
            
        return x


class DoubleConvBlock(nn.Module):
    """
    Double bloc de convolution utilisé dans U-Net original.
    
    Ce bloc applique deux convolutions consécutives avec normalisation
    et activation, comme utilisé dans le U-Net original.
    
    Attributes:
        conv1: Premier bloc de convolution
        conv2: Second bloc de convolution
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU,
        dropout_rate: float = 0.0
    ):
        """
        Initialise un double bloc de convolution.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            mid_channels: Nombre de canaux intermédiaires (si None, égal à out_channels)
            kernel_size: Taille du noyau de convolution
            stride: Pas de la convolution
            padding: Remplissage appliqué aux bords de l'image
            bias: Si True, ajoute un terme de biais
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
            dropout_rate: Taux de dropout (0 pour désactiver)
        """
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
            
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            norm_layer=norm_layer,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        self.conv2 = ConvBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            norm_layer=norm_layer,
            activation=activation,
            dropout_rate=dropout_rate
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du double bloc de convolution.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur de sortie après les deux convolutions
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x 