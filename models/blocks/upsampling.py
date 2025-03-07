"""
Blocs de sur-échantillonnage pour les architectures de réseaux neuronaux.

Ce module fournit différentes implémentations de blocs de sur-échantillonnage
(upsampling) qui peuvent être utilisés dans les architectures d'encodeur-décodeur
comme U-Net pour augmenter la résolution spatiale des caractéristiques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Optional, Tuple, Union

from .conv import ConvBlock


class UpsampleBlock(nn.Module):
    """
    Interface de base pour les blocs de sur-échantillonnage.
    
    Cette classe définit l'interface commune pour tous les blocs
    de sur-échantillonnage et peut être étendue pour créer différentes
    implémentations.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2
    ):
        """
        Initialise un bloc de sur-échantillonnage.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            scale_factor: Facteur d'échelle pour le sur-échantillonnage
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Passage avant du bloc de sur-échantillonnage.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            skip: Tenseur de connexion de saut (skip connection) [B, C', H*scale, W*scale]
            
        Returns:
            Tenseur sur-échantillonné [B, C'', H*scale, W*scale]
        """
        raise NotImplementedError("Les sous-classes doivent implémenter cette méthode")


class TransposeConvUpsampling(UpsampleBlock):
    """
    Bloc de sur-échantillonnage utilisant une convolution transposée.
    
    Ce bloc effectue un sur-échantillonnage en utilisant une convolution
    transposée (déconvolution), qui permet d'apprendre les paramètres
    de sur-échantillonnage.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        kernel_size: int = 4,  # Généralement 4 pour une convolution transposée
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU
    ):
        """
        Initialise un bloc de sur-échantillonnage par convolution transposée.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            scale_factor: Facteur d'échelle pour le sur-échantillonnage
            kernel_size: Taille du noyau de convolution transposée
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
        """
        super().__init__(in_channels, out_channels, scale_factor)
        
        # Le padding et l'output_padding doivent être calculés pour des dimensions correctes
        padding = kernel_size // 4
        output_padding = 0 if scale_factor % 2 == 0 else 1
        
        # Convolution transposée pour le sur-échantillonnage
        self.upconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=scale_factor,
            padding=padding,
            output_padding=output_padding,
            bias=False
        )
        
        # Normalisation et activation
        self.norm = norm_layer(out_channels) if norm_layer else None
        self.activation = activation(inplace=True) if activation else None
        
        # Convolution pour traiter la concaténation avec skip connection
        self.conv_skip = ConvBlock(
            in_channels=out_channels * 2,  # Concaténation de up et skip
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_layer=norm_layer,
            activation=activation
        )
        
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Passage avant du bloc de convolution transposée.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            skip: Tenseur de connexion de saut [B, C', H*scale, W*scale]
            
        Returns:
            Tenseur sur-échantillonné [B, C'', H*scale, W*scale]
        """
        # Sur-échantillonnage avec convolution transposée
        x = self.upconv(x)
        
        if self.norm is not None:
            x = self.norm(x)
            
        if self.activation is not None:
            x = self.activation(x)
            
        # Concaténer avec skip connection si disponible
        if skip is not None:
            # Vérifier que les dimensions spatiales correspondent
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode='bilinear', align_corners=False
                )
            
            x = torch.cat([x, skip], dim=1)
            x = self.conv_skip(x)
            
        return x


class BilinearUpsampling(UpsampleBlock):
    """
    Bloc de sur-échantillonnage utilisant une interpolation bilinéaire.
    
    Ce bloc effectue un sur-échantillonnage en utilisant une interpolation
    bilinéaire suivie d'une convolution pour ajuster le nombre de canaux.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU
    ):
        """
        Initialise un bloc de sur-échantillonnage par interpolation bilinéaire.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            scale_factor: Facteur d'échelle pour le sur-échantillonnage
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
        """
        super().__init__(in_channels, out_channels, scale_factor)
        
        # Convolution 1x1 pour ajuster le nombre de canaux après l'upsampling
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation=activation
        )
        
        # Convolution pour traiter la concaténation avec skip connection
        self.conv_skip = ConvBlock(
            in_channels=out_channels * 2,  # Concaténation de up et skip
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_layer=norm_layer,
            activation=activation
        )
        
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Passage avant du bloc d'interpolation bilinéaire.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            skip: Tenseur de connexion de saut [B, C', H*scale, W*scale]
            
        Returns:
            Tenseur sur-échantillonné [B, C'', H*scale, W*scale]
        """
        # Sur-échantillonnage avec interpolation bilinéaire
        if skip is not None:
            target_size = skip.shape[2:]
        else:
            target_size = (x.shape[2] * self.scale_factor, 
                          x.shape[3] * self.scale_factor)
        
        x = F.interpolate(
            x, size=target_size, mode='bilinear', align_corners=False
        )
        
        # Ajuster le nombre de canaux
        x = self.conv(x)
        
        # Concaténer avec skip connection si disponible
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.conv_skip(x)
            
        return x


class PixelShuffleUpsampling(UpsampleBlock):
    """
    Bloc de sur-échantillonnage utilisant PixelShuffle.
    
    Ce bloc effectue un sur-échantillonnage en utilisant l'opération PixelShuffle,
    qui réorganise les éléments des tenseurs de caractéristiques pour augmenter
    la résolution spatiale.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU
    ):
        """
        Initialise un bloc de sur-échantillonnage par PixelShuffle.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            scale_factor: Facteur d'échelle pour le sur-échantillonnage
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
        """
        super().__init__(in_channels, out_channels, scale_factor)
        
        # Calcul du nombre de canaux nécessaires pour PixelShuffle
        # Pour un facteur d'échelle 's', nous avons besoin de r^2 fois plus de canaux
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels * (scale_factor ** 2),
            kernel_size=1,
            norm_layer=None,  # La normalisation est appliquée après PixelShuffle
            activation=None   # L'activation est appliquée après PixelShuffle
        )
        
        # PixelShuffle pour le sur-échantillonnage
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        # Normalisation et activation après PixelShuffle
        self.norm = norm_layer(out_channels) if norm_layer else None
        self.activation = activation(inplace=True) if activation else None
        
        # Convolution pour traiter la concaténation avec skip connection
        self.conv_skip = ConvBlock(
            in_channels=out_channels * 2,  # Concaténation de up et skip
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_layer=norm_layer,
            activation=activation
        )
        
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Passage avant du bloc PixelShuffle.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            skip: Tenseur de connexion de saut [B, C', H*scale, W*scale]
            
        Returns:
            Tenseur sur-échantillonné [B, C'', H*scale, W*scale]
        """
        # Convolution pour préparer les données pour PixelShuffle
        x = self.conv(x)
        
        # Sur-échantillonnage avec PixelShuffle
        x = self.pixel_shuffle(x)
        
        if self.norm is not None:
            x = self.norm(x)
            
        if self.activation is not None:
            x = self.activation(x)
            
        # Adapter les dimensions spatiales à celles de skip si nécessaire
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode='bilinear', align_corners=False
                )
            
            # Concaténer avec skip connection
            x = torch.cat([x, skip], dim=1)
            x = self.conv_skip(x)
            
        return x 