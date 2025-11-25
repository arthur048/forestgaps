"""
Blocs résiduels avancés avec DropPath, FiLM et CBAM pour les architectures de réseaux neuronaux.

Ce module fournit des implémentations de blocs résiduels avancés qui intègrent
diverses techniques de régularisation et d'attention pour améliorer les performances
des modèles de segmentation d'images forestières.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Optional, Tuple, Union

from .conv import ConvBlock
from .attention import CBAM
from .droppath import DropPath


class ResidualBlockWithDropPath(nn.Module):
    """
    Bloc résiduel avec DropPath pour la régularisation.
    
    Ce bloc implémente un bloc résiduel standard avec l'ajout du mécanisme
    DropPath (Stochastic Depth) pour améliorer la régularisation lors de
    l'entraînement de réseaux profonds.
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
        drop_path_rate: float = 0.0
    ):
        """
        Initialise un bloc résiduel avec DropPath.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            kernel_size: Taille du noyau des convolutions
            stride: Pas des convolutions
            padding: Padding des convolutions (calculé automatiquement si None)
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
            dropout_rate: Taux de dropout classique
            drop_path_rate: Taux de DropPath
        """
        super().__init__()
        
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        # Chemin principal avec deux convolutions
        self.conv1 = ConvBlock(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_layer=norm_layer,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        self.conv2 = ConvBlock(
            out_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            norm_layer=norm_layer,
            activation=None,  # Pas d'activation avant l'addition
            dropout_rate=dropout_rate
        )
        
        # DropPath pour la régularisation
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
        # Connexion résiduelle
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                norm_layer=norm_layer,
                activation=None
            )
        
        # Activation finale
        self.activation = activation() if activation else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagation avant dans le bloc résiduel avec DropPath.
        
        Args:
            x: Tensor d'entrée [B, C, H, W]
            
        Returns:
            Tensor de sortie [B, C_out, H_out, W_out]
        """
        identity = self.shortcut(x)
        
        # Chemin principal
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Appliquer DropPath
        out = self.drop_path(out)
        
        # Addition avec la connexion résiduelle
        out = out + identity
        
        # Activation finale
        out = self.activation(out)
        
        return out


class ResidualBlockWithCBAM(nn.Module):
    """
    Bloc résiduel avec mécanisme d'attention CBAM.
    
    Ce bloc intègre le mécanisme d'attention CBAM (Convolutional Block Attention Module)
    qui combine l'attention par canal et l'attention spatiale pour améliorer
    la capacité du réseau à se concentrer sur les caractéristiques pertinentes.
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
        reduction_ratio: int = 16
    ):
        """
        Initialise un bloc résiduel avec CBAM.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            kernel_size: Taille du noyau des convolutions
            stride: Pas des convolutions
            padding: Padding des convolutions (calculé automatiquement si None)
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
            dropout_rate: Taux de dropout
            reduction_ratio: Ratio de réduction pour le module d'attention
        """
        super().__init__()
        
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        # Chemin principal avec deux convolutions
        self.conv1 = ConvBlock(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_layer=norm_layer,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        self.conv2 = ConvBlock(
            out_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            norm_layer=norm_layer,
            activation=None,  # Pas d'activation avant l'attention
            dropout_rate=dropout_rate
        )
        
        # Module d'attention CBAM
        self.cbam = CBAM(
            channels=out_channels,
            reduction_ratio=reduction_ratio,
            spatial_kernel_size=7,
            activation=activation
        )
        
        # Connexion résiduelle
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                norm_layer=norm_layer,
                activation=None
            )
        
        # Activation finale
        self.activation = activation() if activation else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagation avant dans le bloc résiduel avec CBAM.
        
        Args:
            x: Tensor d'entrée [B, C, H, W]
            
        Returns:
            Tensor de sortie [B, C_out, H_out, W_out]
        """
        identity = self.shortcut(x)
        
        # Chemin principal
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Appliquer l'attention CBAM
        out = self.cbam(out)
        
        # Addition avec la connexion résiduelle
        out = out + identity
        
        # Activation finale
        out = self.activation(out)
        
        return out


class FiLMResidualBlock(nn.Module):
    """
    Bloc résiduel avec modulation FiLM.
    
    Ce bloc intègre la modulation FiLM (Feature-wise Linear Modulation) qui
    permet de conditionner les caractéristiques du réseau en fonction d'un
    paramètre externe, comme un seuil de hauteur.
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
        dropout_rate: float = 0.0
    ):
        """
        Initialise un bloc résiduel avec FiLM.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            kernel_size: Taille du noyau des convolutions
            stride: Pas des convolutions
            padding: Padding des convolutions (calculé automatiquement si None)
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
            dropout_rate: Taux de dropout
        """
        super().__init__()
        
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        # Chemin principal avec deux convolutions
        self.conv1 = ConvBlock(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_layer=norm_layer,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        self.conv2 = ConvBlock(
            out_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            norm_layer=norm_layer,
            activation=None,  # Pas d'activation avant l'addition
            dropout_rate=dropout_rate
        )
        
        # Connexion résiduelle
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                norm_layer=norm_layer,
                activation=None
            )
        
        # Activation finale
        self.activation = activation() if activation else nn.Identity()
        
    def forward(self, x: torch.Tensor, gamma: Optional[torch.Tensor] = None, beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Propagation avant dans le bloc résiduel avec FiLM.
        
        Args:
            x: Tensor d'entrée [B, C, H, W]
            gamma: Paramètre de modulation multiplicatif [B, C] ou None
            beta: Paramètre de modulation additif [B, C] ou None
            
        Returns:
            Tensor de sortie [B, C_out, H_out, W_out]
        """
        identity = self.shortcut(x)
        
        # Chemin principal
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Appliquer la modulation FiLM si les paramètres sont fournis
        if gamma is not None and beta is not None:
            # Adapter les dimensions pour le broadcasting
            gamma = gamma.view(-1, gamma.size(1), 1, 1)
            beta = beta.view(-1, beta.size(1), 1, 1)
            
            # Appliquer la modulation
            out = gamma * out + beta
        
        # Addition avec la connexion résiduelle
        out = out + identity
        
        # Activation finale
        out = self.activation(out)
        
        return out


class ResidualBlockWithFiLMCBAMDropPath(nn.Module):
    """
    Bloc résiduel combinant FiLM, CBAM et DropPath.
    
    Ce bloc avancé intègre toutes les fonctionnalités:
    - Modulation FiLM pour le conditionnement par des paramètres externes
    - Attention CBAM pour cibler les caractéristiques pertinentes
    - DropPath pour la régularisation des réseaux profonds
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
        drop_path_rate: float = 0.0,
        reduction_ratio: int = 16
    ):
        """
        Initialise un bloc résiduel avec FiLM, CBAM et DropPath.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            kernel_size: Taille du noyau des convolutions
            stride: Pas des convolutions
            padding: Padding des convolutions (calculé automatiquement si None)
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
            dropout_rate: Taux de dropout classique
            drop_path_rate: Taux de DropPath
            reduction_ratio: Ratio de réduction pour le module CBAM
        """
        super().__init__()
        
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        # Chemin principal avec deux convolutions
        self.conv1 = ConvBlock(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_layer=norm_layer,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        self.conv2 = ConvBlock(
            out_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            norm_layer=norm_layer,
            activation=None,  # Pas d'activation avant les modules suivants
            dropout_rate=dropout_rate
        )
        
        # Module d'attention CBAM
        self.cbam = CBAM(
            channels=out_channels,
            reduction_ratio=reduction_ratio,
            spatial_kernel_size=7,
            activation=activation
        )
        
        # DropPath pour la régularisation
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
        # Connexion résiduelle
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                norm_layer=norm_layer,
                activation=None
            )
        
        # Activation finale
        self.activation = activation() if activation else nn.Identity()
        
    def forward(self, x: torch.Tensor, gamma: Optional[torch.Tensor] = None, beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Propagation avant dans le bloc résiduel avec FiLM, CBAM et DropPath.
        
        Args:
            x: Tensor d'entrée [B, C, H, W]
            gamma: Paramètre de modulation multiplicatif [B, C] ou None
            beta: Paramètre de modulation additif [B, C] ou None
            
        Returns:
            Tensor de sortie [B, C_out, H_out, W_out]
        """
        identity = self.shortcut(x)
        
        # Chemin principal
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Appliquer la modulation FiLM si les paramètres sont fournis
        if gamma is not None and beta is not None:
            # Adapter les dimensions pour le broadcasting
            gamma = gamma.view(-1, gamma.size(1), 1, 1)
            beta = beta.view(-1, beta.size(1), 1, 1)
            
            # Appliquer la modulation
            out = gamma * out + beta
        
        # Appliquer l'attention CBAM
        out = self.cbam(out)
        
        # Appliquer DropPath
        out = self.drop_path(out)
        
        # Addition avec la connexion résiduelle
        out = out + identity
        
        # Activation finale
        out = self.activation(out)
        
        return out 