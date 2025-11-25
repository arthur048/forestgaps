"""
Blocs de convolution pour les architectures de réseaux neuronaux.

Ce module fournit différents blocs de convolution réutilisables
qui peuvent être utilisés comme composants dans diverses architectures
de réseaux neuronaux pour la segmentation d'images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Optional, Tuple, List, Union


class ConvBlock(nn.Module):
    """
    Bloc de convolution simple avec normalisation et activation.
    
    Ce bloc comprend une convolution 2D suivie d'une normalisation
    et d'une activation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        dropout_rate: float = 0.0
    ):
        """
        Initialise un bloc de convolution.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            kernel_size: Taille du noyau de convolution
            stride: Pas de la convolution
            padding: Remplissage autour de l'entrée (si None, padding = kernel_size // 2)
            dilation: Espacement entre les éléments du noyau
            groups: Nombre de connexions bloquées
            bias: Si True, ajoute un terme de biais à la sortie
            padding_mode: Type de remplissage ('zeros', 'reflect', 'replicate', 'circular')
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
            dropout_rate: Taux de dropout (0 pour désactiver)
        """
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
            
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )
        
        # Ajouter une couche de normalisation si spécifiée
        self.norm = norm_layer(out_channels) if norm_layer else None
        
        # Ajouter une fonction d'activation si spécifiée
        self.activation = activation(inplace=True) if activation else None
        
        # Ajouter un dropout si spécifié
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
        
        if self.norm is not None:
            x = self.norm(x)
            
        if self.activation is not None:
            x = self.activation(x)
            
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x


class DoubleConvBlock(nn.Module):
    """
    Bloc à double convolution avec normalisation et activation.
    
    Ce bloc comprend deux convolutions 2D consécutives, chacune suivie
    d'une normalisation et d'une activation. Ce type de bloc est couramment
    utilisé dans l'architecture U-Net.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        dropout_rate: float = 0.0
    ):
        """
        Initialise un bloc à double convolution.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            mid_channels: Nombre de canaux intermédiaires (si None, utilise out_channels)
            kernel_size: Taille du noyau de convolution
            stride: Pas de la convolution
            padding: Remplissage autour de l'entrée (si None, padding = kernel_size // 2)
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
            dropout_rate: Taux de dropout
        """
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
            
        # Première convolution
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_layer=norm_layer,
            activation=activation,
            dropout_rate=0.0  # Dropout seulement après la dernière conv
        )
        
        # Seconde convolution
        self.conv2 = ConvBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,  # Toujours 1 pour la seconde conv
            padding=padding,
            norm_layer=norm_layer,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc à double convolution.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur de sortie après les deux convolutions
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResidualBlock(nn.Module):
    """
    Bloc résiduel avec connexion de saut.
    
    Ce bloc implémente une connexion résiduelle similaire à celle
    des architectures ResNet, permettant un meilleur flux du gradient
    à travers les couches profondes.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        dropout_rate: float = 0.0,
        downsample: bool = False
    ):
        """
        Initialise un bloc résiduel.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            kernel_size: Taille du noyau de convolution
            stride: Pas de la convolution
            padding: Remplissage autour de l'entrée (si None, padding = kernel_size // 2)
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
            dropout_rate: Taux de dropout
            downsample: Si True, effectue un sous-échantillonnage spatial avec stride=2
        """
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        # Appliquer stride=2 au premier bloc si downsample est True
        actual_stride = 2 if downsample else stride
        
        # Premier bloc de convolution
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=actual_stride,
            padding=padding,
            norm_layer=norm_layer,
            activation=activation,
            dropout_rate=0.0
        )
        
        # Second bloc de convolution sans stride
        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            norm_layer=norm_layer,
            activation=None,  # L'activation est appliquée après l'addition
            dropout_rate=0.0
        )
        
        # Connexion de saut (shortcut)
        self.shortcut = nn.Identity()
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=actual_stride, bias=False),
                norm_layer(out_channels) if norm_layer else nn.Identity()
            )
        
        # Activation après l'addition
        self.activation = activation(inplace=True) if activation else nn.Identity()
        
        # Dropout à la fin
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc résiduel.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur de sortie après les convolutions et l'addition résiduelle
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        out += identity
        out = self.activation(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
            
        return out


class SEBlock(nn.Module):
    """
    Bloc Squeeze-and-Excitation (SE).
    
    Ce bloc implémente un mécanisme d'attention par canal qui recalibre
    adaptativement les caractéristiques des canaux en modélisant explicitement
    les interdépendances entre les canaux.
    """
    
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        activation: Type[nn.Module] = nn.ReLU
    ):
        """
        Initialise un bloc Squeeze-and-Excitation.
        
        Args:
            channels: Nombre de canaux d'entrée
            reduction_ratio: Facteur de réduction pour le goulot d'étranglement
            activation: Fonction d'activation à utiliser
        """
        super().__init__()
        
        # Assurer que le nombre de canaux après réduction est au moins 1
        reduced_channels = max(1, channels // reduction_ratio)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            activation(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc SE.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur de sortie avec attention par canal
        """
        batch_size, channels, _, _ = x.size()
        
        # Squeeze: réduction spatiale globale
        y = self.avg_pool(x).view(batch_size, channels)
        
        # Excitation: recalibrage des canaux
        y = self.fc(y).view(batch_size, channels, 1, 1)
        
        # Multiplication par les poids d'attention
        return x * y 