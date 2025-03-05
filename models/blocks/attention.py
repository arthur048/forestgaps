"""
Mécanismes d'attention pour les architectures de réseaux neuronaux.

Ce module fournit différentes implémentations de mécanismes d'attention
qui peuvent être intégrés dans les architectures de réseaux neuronaux
pour améliorer leur capacité à se concentrer sur les caractéristiques pertinentes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Optional, Tuple, Union

from .conv import ConvBlock


class AttentionGate(nn.Module):
    """
    Porte d'attention (Attention Gate) pour U-Net.
    
    Cette implémentation est basée sur l'article "Attention U-Net" et permet
    de se concentrer sur les régions pertinentes en utilisant des signaux de
    guidage provenant des caractéristiques de niveau supérieur.
    """
    
    def __init__(
        self,
        g_channels: int,  # Canaux du signal de guidage (gating signal)
        x_channels: int,  # Canaux de la caractéristique d'entrée
        inter_channels: Optional[int] = None,  # Canaux intermédiaires
        activation: Type[nn.Module] = nn.ReLU
    ):
        """
        Initialise une porte d'attention.
        
        Args:
            g_channels: Nombre de canaux du signal de guidage
            x_channels: Nombre de canaux de la caractéristique d'entrée
            inter_channels: Nombre de canaux intermédiaires
            activation: Fonction d'activation à utiliser
        """
        super().__init__()
        
        if inter_channels is None:
            inter_channels = x_channels // 2
            
        # Réduction des dimensions pour le calcul d'attention
        self.W_g = nn.Conv2d(
            g_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        
        self.W_x = nn.Conv2d(
            x_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        
        self.psi = nn.Conv2d(
            inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True
        )
        
        self.activation = activation(inplace=True) if activation else nn.Identity()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant de la porte d'attention.
        
        Args:
            g: Signal de guidage [B, g_channels, H', W']
            x: Caractéristique d'entrée [B, x_channels, H, W]
            
        Returns:
            Caractéristique recalibrée par l'attention [B, x_channels, H, W]
        """
        # Adapter les dimensions du signal de guidage à celles de x
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        # Calcul des projections
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Activation des projections additionnées
        psi = self.activation(g1 + x1)
        
        # Calcul des coefficients d'attention
        psi = self.psi(psi)
        alpha = self.sigmoid(psi)
        
        # Application de l'attention
        return x * alpha


class SpatialAttentionBlock(nn.Module):
    """
    Bloc d'attention spatiale.
    
    Ce bloc implémente un mécanisme d'attention qui se concentre sur
    les régions spatiales importantes de l'entrée.
    """
    
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 8,
        kernel_size: int = 7
    ):
        """
        Initialise un bloc d'attention spatiale.
        
        Args:
            channels: Nombre de canaux d'entrée/sortie
            reduction_ratio: Facteur de réduction pour le goulot d'étranglement
            kernel_size: Taille du noyau pour la convolution spatiale
        """
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Assurer que le kernel_size est impair pour un padding symétrique
        assert kernel_size % 2 == 1, "kernel_size doit être impair"
        padding = kernel_size // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc d'attention spatiale.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur recalibré spatialement [B, C, H, W]
        """
        # Générer des cartes de caractéristiques moyennes et maximales le long des canaux
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concaténer les deux cartes de caractéristiques
        y = torch.cat([avg_out, max_out], dim=1)
        
        # Générer la carte d'attention
        y = self.conv(y)
        
        # Appliquer l'attention
        return x * y


class ChannelAttentionBlock(nn.Module):
    """
    Bloc d'attention par canal.
    
    Ce bloc implémente un mécanisme d'attention qui se concentre sur
    les canaux importants de l'entrée, similaire au bloc SE mais avec
    une architecture différente.
    """
    
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        activation: Type[nn.Module] = nn.ReLU
    ):
        """
        Initialise un bloc d'attention par canal.
        
        Args:
            channels: Nombre de canaux d'entrée/sortie
            reduction_ratio: Facteur de réduction pour le goulot d'étranglement
            activation: Fonction d'activation à utiliser
        """
        super().__init__()
        
        # Assurer que le nombre de canaux après réduction est au moins 1
        reduced_channels = max(1, channels // reduction_ratio)
        
        # Branche avec pooling moyen
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            activation(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False)
        )
        
        # Branche avec pooling max
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            activation(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False)
        )
        
        # Fonction d'activation finale
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc d'attention par canal.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur recalibré par canal [B, C, H, W]
        """
        batch_size, channels, _, _ = x.size()
        
        # Branche moyenne
        avg_pool = self.avg_pool(x).view(batch_size, channels)
        avg_attention = self.avg_fc(avg_pool)
        
        # Branche max
        max_pool = self.max_pool(x).view(batch_size, channels)
        max_attention = self.max_fc(max_pool)
        
        # Fusionner les deux branches
        attention = self.sigmoid(avg_attention + max_attention).view(batch_size, channels, 1, 1)
        
        # Appliquer l'attention
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Ce module combine l'attention par canal et l'attention spatiale
    pour améliorer la représentation des caractéristiques.
    """
    
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
        activation: Type[nn.Module] = nn.ReLU
    ):
        """
        Initialise un module CBAM.
        
        Args:
            channels: Nombre de canaux d'entrée/sortie
            reduction_ratio: Facteur de réduction pour l'attention par canal
            spatial_kernel_size: Taille du noyau pour l'attention spatiale
            activation: Fonction d'activation à utiliser
        """
        super().__init__()
        
        # Attention par canal
        self.channel_attention = ChannelAttentionBlock(
            channels=channels,
            reduction_ratio=reduction_ratio,
            activation=activation
        )
        
        # Attention spatiale
        self.spatial_attention = SpatialAttentionBlock(
            channels=channels,
            kernel_size=spatial_kernel_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du module CBAM.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur recalibré [B, C, H, W]
        """
        # Appliquer l'attention par canal
        x = self.channel_attention(x)
        
        # Appliquer l'attention spatiale
        x = self.spatial_attention(x)
        
        return x 