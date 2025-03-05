"""
Module d'auto-attention (Self-Attention) pour les architectures U-Net.

Ce module implémente le mécanisme d'auto-attention qui permet au modèle
de capturer des dépendances à longue distance dans les images.

Référence: Wang, X., Girshick, R., Gupta, A., & He, K. (2018).
Non-local Neural Networks. CVPR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Module d'auto-attention pour les réseaux de convolution.
    
    Ce module implémente un mécanisme d'auto-attention qui permet
    au modèle de capturer des dépendances à longue distance dans les images.
    
    Attributes:
        query_conv: Convolution pour générer les requêtes
        key_conv: Convolution pour générer les clés
        value_conv: Convolution pour générer les valeurs
        gamma: Paramètre appris pour pondérer l'importance de l'attention
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        """
        Initialise le module d'auto-attention.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            reduction_ratio: Ratio de réduction pour les projections
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.reduced_channels = in_channels // reduction_ratio
        
        self.query_conv = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du module d'auto-attention.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur de sortie après application de l'auto-attention
        """
        batch_size, C, height, width = x.size()
        
        # Générer les requêtes, clés et valeurs
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x (H*W) x C'
        key = self.key_conv(x).view(batch_size, -1, height * width)  # B x C' x (H*W)
        value = self.value_conv(x).view(batch_size, -1, height * width)  # B x C x (H*W)
        
        # Calculer la matrice d'attention
        attention = torch.bmm(query, key)  # B x (H*W) x (H*W)
        attention = F.softmax(attention, dim=-1)
        
        # Appliquer l'attention aux valeurs
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, C, height, width)
        
        # Combiner avec l'entrée originale
        out = self.gamma * out + x
        
        return out


class PositionAttention(nn.Module):
    """
    Module d'attention de position pour les réseaux de segmentation.
    
    Ce module implémente un mécanisme d'attention qui se concentre
    sur les relations spatiales entre les pixels.
    
    Attributes:
        query_conv: Convolution pour générer les requêtes
        key_conv: Convolution pour générer les clés
        value_conv: Convolution pour générer les valeurs
    """
    
    def __init__(self, in_channels: int):
        """
        Initialise le module d'attention de position.
        
        Args:
            in_channels: Nombre de canaux d'entrée
        """
        super().__init__()
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du module d'attention de position.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur de sortie après application de l'attention de position
        """
        batch_size, C, height, width = x.size()
        
        # Générer les requêtes, clés et valeurs
        query = self.query_conv(x)  # B x C' x H x W
        key = self.key_conv(x)  # B x C' x H x W
        value = self.value_conv(x)  # B x C x H x W
        
        # Reshape pour le calcul de l'attention
        query_flat = query.view(batch_size, -1, height * width).permute(0, 2, 1)  # B x (H*W) x C'
        key_flat = key.view(batch_size, -1, height * width)  # B x C' x (H*W)
        value_flat = value.view(batch_size, -1, height * width)  # B x C x (H*W)
        
        # Calculer la matrice d'attention
        attention = torch.bmm(query_flat, key_flat)  # B x (H*W) x (H*W)
        attention = F.softmax(attention, dim=-1)
        
        # Appliquer l'attention aux valeurs
        out = torch.bmm(value_flat, attention.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, C, height, width)
        
        # Combiner avec l'entrée originale
        out = self.gamma * out + x
        
        return out 