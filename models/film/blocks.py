"""
Blocs FiLM pour les architectures U-Net.

Ce module fournit des implémentations de blocs qui intègrent
les couches FiLM dans les architectures U-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.convolution import ConvBlock
from models.blocks.residual import ResidualBlock
from models.film.layers import FiLMLayer, FiLMGenerator


class FiLMBlock(nn.Module):
    """
    Bloc de convolution avec modulation FiLM.
    
    Ce bloc combine une couche de convolution avec une modulation FiLM
    pour conditionner les caractéristiques en fonction de paramètres externes.
    
    Attributes:
        conv: Bloc de convolution
        film: Couche FiLM
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU
    ):
        """
        Initialise un bloc FiLM.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            kernel_size: Taille du noyau de convolution
            stride: Pas de la convolution
            padding: Remplissage appliqué aux bords de l'image
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
        """
        super().__init__()
        
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_layer=norm_layer,
            activation=None  # L'activation sera appliquée après FiLM
        )
        
        self.film = FiLMLayer(num_features=out_channels)
        self.activation = activation(inplace=True) if activation else None
        
    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc FiLM.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            gamma: Paramètres de mise à l'échelle [B, C]
            beta: Paramètres de décalage [B, C]
            
        Returns:
            Tenseur modulé
        """
        x = self.conv(x)
        x = self.film(x, gamma, beta)
        
        if self.activation:
            x = self.activation(x)
            
        return x


class FiLMResidualBlock(nn.Module):
    """
    Bloc résiduel avec modulation FiLM.
    
    Ce bloc combine un bloc résiduel avec une modulation FiLM
    pour conditionner les caractéristiques en fonction de paramètres externes.
    
    Attributes:
        residual: Bloc résiduel
        film: Couche FiLM
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU,
        downsample: bool = False
    ):
        """
        Initialise un bloc résiduel FiLM.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            stride: Pas de la convolution
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
            downsample: Si True, réduit la résolution spatiale (stride=2)
        """
        super().__init__()
        
        # Utiliser un bloc résiduel sans activation finale
        self.residual = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            norm_layer=norm_layer,
            activation=None,  # Pas d'activation finale
            downsample=downsample
        )
        
        self.film = FiLMLayer(num_features=out_channels)
        self.activation = activation(inplace=True) if activation else None
        
    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc résiduel FiLM.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            gamma: Paramètres de mise à l'échelle [B, C]
            beta: Paramètres de décalage [B, C]
            
        Returns:
            Tenseur modulé
        """
        x = self.residual(x)
        x = self.film(x, gamma, beta)
        
        if self.activation:
            x = self.activation(x)
            
        return x


class ConditionedBlock(nn.Module):
    """
    Bloc conditionné par un vecteur externe.
    
    Ce bloc intègre un générateur FiLM et un bloc FiLM pour
    conditionner les caractéristiques en fonction d'un vecteur externe.
    
    Attributes:
        generator: Générateur de paramètres FiLM
        film_block: Bloc FiLM
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conditioning_size: int,
        use_residual: bool = False,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU
    ):
        """
        Initialise un bloc conditionné.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            conditioning_size: Taille du vecteur de conditionnement
            use_residual: Si True, utilise un bloc résiduel
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
        """
        super().__init__()
        
        self.generator = FiLMGenerator(
            conditioning_size=conditioning_size,
            num_features=out_channels
        )
        
        if use_residual:
            self.film_block = FiLMResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_layer=norm_layer,
                activation=activation
            )
        else:
            self.film_block = FiLMBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_layer=norm_layer,
                activation=activation
            )
        
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc conditionné.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            conditioning: Vecteur de conditionnement [B, conditioning_size]
            
        Returns:
            Tenseur modulé
        """
        gamma, beta = self.generator(conditioning)
        return self.film_block(x, gamma, beta) 