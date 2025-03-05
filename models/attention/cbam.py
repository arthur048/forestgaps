"""
Convolutional Block Attention Module (CBAM).

Ce module implémente le mécanisme d'attention CBAM qui combine
l'attention spatiale et l'attention des canaux pour améliorer
les performances des réseaux de convolution.

Référence: Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018).
CBAM: Convolutional Block Attention Module. ECCV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Module d'attention des canaux.
    
    Ce module calcule des poids d'attention pour chaque canal
    en utilisant à la fois le pooling moyen et le pooling max.
    
    Attributes:
        avg_pool: Couche de pooling moyen global
        max_pool: Couche de pooling max global
        mlp: Réseau MLP pour calculer les poids d'attention
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        """
        Initialise le module d'attention des canaux.
        
        Args:
            channels: Nombre de canaux d'entrée
            reduction_ratio: Ratio de réduction pour le MLP
        """
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du module d'attention des canaux.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur d'attention des canaux [B, C, 1, 1]
        """
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Module d'attention spatiale.
    
    Ce module calcule des poids d'attention pour chaque position spatiale
    en utilisant à la fois les caractéristiques moyennes et maximales des canaux.
    
    Attributes:
        conv: Couche de convolution pour calculer les poids d'attention
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Initialise le module d'attention spatiale.
        
        Args:
            kernel_size: Taille du noyau de convolution
        """
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du module d'attention spatiale.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur d'attention spatiale [B, 1, H, W]
        """
        # Calculer les caractéristiques moyennes et maximales des canaux
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concaténer les caractéristiques
        out = torch.cat([avg_out, max_out], dim=1)
        
        # Appliquer la convolution et la fonction sigmoid
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Ce module combine l'attention des canaux et l'attention spatiale
    pour améliorer les performances des réseaux de convolution.
    
    Attributes:
        channel_attention: Module d'attention des canaux
        spatial_attention: Module d'attention spatiale
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7):
        """
        Initialise le module CBAM.
        
        Args:
            channels: Nombre de canaux d'entrée
            reduction_ratio: Ratio de réduction pour l'attention des canaux
            spatial_kernel_size: Taille du noyau pour l'attention spatiale
        """
        super().__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du module CBAM.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur de sortie après application de l'attention
        """
        # Appliquer l'attention des canaux
        x = x * self.channel_attention(x)
        
        # Appliquer l'attention spatiale
        x = x * self.spatial_attention(x)
        
        return x 