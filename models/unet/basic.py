"""
Implémentation de base de l'architecture U-Net.

Ce module fournit une implémentation de l'architecture U-Net originale
pour la segmentation d'images de télédétection forestière.

Référence: Ronneberger, O., Fischer, P., & Brox, T. (2015).
U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union

from models.blocks.convolution import DoubleConvBlock
from models.blocks.pooling import DownsampleBlock, UpsampleBlock


class UNet(nn.Module):
    """
    Architecture U-Net de base pour la segmentation d'images.
    
    Cette classe implémente l'architecture U-Net originale avec
    un encodeur, un décodeur et des connexions de saut.
    
    Attributes:
        encoder: Modules de l'encodeur
        bottleneck: Module du goulot d'étranglement
        decoder: Modules du décodeur
        final_conv: Convolution finale pour la prédiction
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.0,
        use_sigmoid: bool = True,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU
    ):
        """
        Initialise le modèle U-Net.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie (classes)
            init_features: Nombre de caractéristiques initial
            depth: Profondeur du réseau (nombre de niveaux)
            dropout_rate: Taux de dropout
            use_sigmoid: Si True, applique une fonction sigmoid à la sortie
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
        """
        super().__init__()
        
        self.depth = depth
        self.use_sigmoid = use_sigmoid
        
        # Encodeur
        self.encoder = nn.ModuleList()
        
        # Premier niveau de l'encodeur
        self.encoder.append(
            DoubleConvBlock(
                in_channels=in_channels,
                out_channels=init_features,
                norm_layer=norm_layer,
                activation=activation,
                dropout_rate=dropout_rate
            )
        )
        
        # Niveaux suivants de l'encodeur
        for i in range(1, depth):
            in_feats = init_features * (2 ** (i - 1))
            out_feats = init_features * (2 ** i)
            
            self.encoder.append(
                nn.Sequential(
                    DownsampleBlock(
                        in_channels=in_feats,
                        pool_type='max'
                    ),
                    DoubleConvBlock(
                        in_channels=in_feats,
                        out_channels=out_feats,
                        norm_layer=norm_layer,
                        activation=activation,
                        dropout_rate=dropout_rate
                    )
                )
            )
        
        # Goulot d'étranglement (bottleneck)
        bottleneck_feats = init_features * (2 ** depth)
        self.bottleneck = nn.Sequential(
            DownsampleBlock(
                in_channels=init_features * (2 ** (depth - 1)),
                pool_type='max'
            ),
            DoubleConvBlock(
                in_channels=init_features * (2 ** (depth - 1)),
                out_channels=bottleneck_feats,
                norm_layer=norm_layer,
                activation=activation,
                dropout_rate=dropout_rate
            )
        )
        
        # Décodeur
        self.decoder = nn.ModuleList()
        
        # Niveaux du décodeur
        for i in range(depth):
            in_feats = init_features * (2 ** (depth - i))
            out_feats = init_features * (2 ** (depth - i - 1))
            
            self.decoder.append(
                nn.ModuleDict({
                    'upsample': UpsampleBlock(
                        in_channels=in_feats,
                        out_channels=out_feats,
                        mode='bilinear',
                        with_conv=True,
                        norm_layer=norm_layer,
                        activation=activation
                    ),
                    'conv': DoubleConvBlock(
                        in_channels=in_feats,  # in_feats = out_feats (upsample) + out_feats (skip)
                        out_channels=out_feats,
                        norm_layer=norm_layer,
                        activation=activation,
                        dropout_rate=dropout_rate
                    )
                })
            )
        
        # Convolution finale
        self.final_conv = nn.Conv2d(
            in_channels=init_features,
            out_channels=out_channels,
            kernel_size=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du modèle U-Net.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur de sortie (segmentation) [B, out_channels, H, W]
        """
        # Stocker les sorties de l'encodeur pour les connexions de saut
        encoder_outputs = []
        
        # Encodeur
        for encoder_block in self.encoder:
            x = encoder_block(x)
            encoder_outputs.append(x)
        
        # Goulot d'étranglement
        x = self.bottleneck(x)
        
        # Décodeur avec connexions de saut
        for i, decoder_block in enumerate(self.decoder):
            skip = encoder_outputs[-(i + 1)]
            x = decoder_block['upsample'](x)
            
            # Redimensionner si nécessaire
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                
            # Concaténer avec la connexion de saut
            x = torch.cat([x, skip], dim=1)
            
            # Appliquer les convolutions
            x = decoder_block['conv'](x)
        
        # Convolution finale
        x = self.final_conv(x)
        
        # Appliquer sigmoid si demandé
        if self.use_sigmoid:
            x = torch.sigmoid(x)
            
        return x 