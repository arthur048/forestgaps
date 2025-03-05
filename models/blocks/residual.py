"""
Blocs résiduels pour les architectures U-Net.

Ce module fournit des implémentations de blocs résiduels qui sont utilisés
dans les architectures U-Net améliorées, comme ResUNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.convolution import ConvBlock


class ResidualBlock(nn.Module):
    """
    Bloc résiduel standard avec connexion de contournement.
    
    Ce bloc implémente la structure standard d'un bloc résiduel avec
    deux couches de convolution et une connexion de contournement.
    
    Attributes:
        conv1: Première couche de convolution
        conv2: Deuxième couche de convolution
        shortcut: Connexion de contournement (identity ou conv1x1)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU,
        dropout_rate: float = 0.0,
        downsample: bool = False
    ):
        """
        Initialise un bloc résiduel.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            stride: Pas de la convolution
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
            dropout_rate: Taux de dropout (0 pour désactiver)
            downsample: Si True, réduit la résolution spatiale (stride=2)
        """
        super().__init__()
        
        stride = 2 if downsample else stride
        
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm_layer=norm_layer,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_layer=norm_layer,
            activation=None,  # No activation before skip connection
            dropout_rate=dropout_rate
        )
        
        # Shortcut connection
        if in_channels != out_channels or stride != 1:
            self.shortcut = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                norm_layer=norm_layer,
                activation=None  # No activation in shortcut
            )
        else:
            self.shortcut = nn.Identity()
            
        self.activation = activation(inplace=True) if activation else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc résiduel.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur de sortie après les convolutions et l'addition de la connexion résiduelle
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        out += identity
        
        if self.activation:
            out = self.activation(out)
            
        return out


class BottleneckBlock(nn.Module):
    """
    Bloc bottleneck avec trois couches de convolution (1x1, 3x3, 1x1).
    
    Ce bloc implémente un bloc résiduel de type bottleneck comme utilisé dans ResNet-50+.
    Il compresse les canaux avec une conv 1x1, puis applique une conv 3x3, puis
    étend à nouveau les canaux avec une autre conv 1x1.
    
    Attributes:
        conv1: Première couche de convolution (1x1, réduction des canaux)
        conv2: Deuxième couche de convolution (3x3)
        conv3: Troisième couche de convolution (1x1, augmentation des canaux)
        shortcut: Connexion de contournement (identity ou conv1x1)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        stride: int = 1,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU,
        dropout_rate: float = 0.0,
        downsample: bool = False,
        expansion: int = 4
    ):
        """
        Initialise un bloc bottleneck.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie (avant expansion)
            mid_channels: Nombre de canaux intermédiaires (si None, égal à out_channels)
            stride: Pas de la convolution
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
            dropout_rate: Taux de dropout (0 pour désactiver)
            downsample: Si True, réduit la résolution spatiale (stride=2)
            expansion: Facteur d'expansion pour les canaux de sortie
        """
        super().__init__()
        
        stride = 2 if downsample else stride
        if mid_channels is None:
            mid_channels = out_channels
            
        expanded_channels = out_channels * expansion
        
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=norm_layer,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        self.conv2 = ConvBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm_layer=norm_layer,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        self.conv3 = ConvBlock(
            in_channels=mid_channels,
            out_channels=expanded_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=norm_layer,
            activation=None,  # No activation before skip connection
            dropout_rate=dropout_rate
        )
        
        # Shortcut connection
        if in_channels != expanded_channels or stride != 1:
            self.shortcut = ConvBlock(
                in_channels=in_channels,
                out_channels=expanded_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                norm_layer=norm_layer,
                activation=None  # No activation in shortcut
            )
        else:
            self.shortcut = nn.Identity()
            
        self.activation = activation(inplace=True) if activation else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc bottleneck.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur de sortie après les convolutions et l'addition de la connexion résiduelle
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        out += identity
        
        if self.activation:
            out = self.activation(out)
            
        return out 