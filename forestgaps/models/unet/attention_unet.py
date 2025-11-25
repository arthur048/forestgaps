"""
Implémentation de l'architecture Attention U-Net pour la segmentation d'images forestières.

Ce module fournit une implémentation de l'architecture Attention U-Net
telle que décrite dans le papier de Oktay et al. (2018), adaptée
pour la détection des trouées forestières.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Type, Union

from ..base import UNetBaseModel
from ..registry import model_registry
from ..blocks.conv import DoubleConvBlock
from ..blocks.downsampling import MaxPoolDownsample
from ..blocks.upsampling import BilinearUpsampling
from ..blocks.attention import AttentionGate


@model_registry.register("attention_unet")
class AttentionUNet(UNetBaseModel):
    """
    Implémentation de l'architecture Attention U-Net.
    
    Cette classe implémente l'architecture Attention U-Net telle que décrite dans
    le papier "Attention U-Net: Learning Where to Look for the Pancreas"
    par Oktay et al. (2018), adaptée pour la détection des trouées forestières.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.0,
        use_sigmoid: bool = True,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU
    ):
        """
        Initialise le modèle Attention U-Net.
        
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
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            depth=depth,
            dropout_rate=dropout_rate,
            use_sigmoid=use_sigmoid,
            norm_layer=norm_layer,
            activation=activation
        )
        
        # Initialiser les listes pour stocker les couches
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        # Construire l'encodeur
        features = init_features
        encoder_features_channels = [features]
        
        # Premier bloc d'encodeur (pas de downsampling)
        self.encoder_blocks.append(
            DoubleConvBlock(
                in_channels=in_channels,
                out_channels=features,
                norm_layer=norm_layer,
                activation=activation
            )
        )
        
        # Blocs d'encodeur restants avec downsampling
        for i in range(depth - 1):
            in_features = features
            features *= 2  # Doubler le nombre de caractéristiques à chaque niveau
            
            # Bloc de sous-échantillonnage
            self.downsample_blocks.append(
                MaxPoolDownsample(
                    in_channels=in_features,
                    out_channels=in_features,
                    scale_factor=2
                )
            )
            
            # Bloc de convolution après sous-échantillonnage
            self.encoder_blocks.append(
                DoubleConvBlock(
                    in_channels=in_features,
                    out_channels=features,
                    norm_layer=norm_layer,
                    activation=activation,
                    dropout_rate=dropout_rate if i == depth - 2 else 0.0
                )
            )
            
            encoder_features_channels.append(features)
        
        # Goulot d'étranglement (bottleneck)
        self.bottleneck = DoubleConvBlock(
            in_channels=features,
            out_channels=features * 2,
            norm_layer=norm_layer,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        # Construire le décodeur
        bottleneck_features = features * 2
        features = bottleneck_features
        
        # Blocs de décodeur
        for i in range(depth):
            in_features = features
            out_features = encoder_features_channels[-(i+1)]
            
            # Porte d'attention
            self.attention_gates.append(
                AttentionGate(
                    g_channels=in_features,  # Signal de guidage du niveau supérieur
                    x_channels=out_features,  # Caractéristiques de l'encodeur
                    inter_channels=out_features // 2,  # Canaux intermédiaires
                    activation=activation
                )
            )
            
            # Bloc de sur-échantillonnage
            self.upsample_blocks.append(
                BilinearUpsampling(
                    in_channels=in_features,
                    out_channels=out_features,
                    scale_factor=2,
                    norm_layer=norm_layer,
                    activation=activation
                )
            )
            
            # Bloc de convolution après sur-échantillonnage et concaténation
            self.decoder_blocks.append(
                DoubleConvBlock(
                    in_channels=out_features * 2,  # Concaténation avec skip
                    out_channels=out_features,
                    norm_layer=norm_layer,
                    activation=activation,
                    dropout_rate=dropout_rate if i < 2 else 0.0
                )
            )
            
            features = out_features
        
        # Couche de convolution finale
        self.final_conv = nn.Conv2d(
            in_channels=features,
            out_channels=out_channels,
            kernel_size=1
        )
        
        # Fonction d'activation finale
        self.sigmoid = nn.Sigmoid() if use_sigmoid else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du modèle Attention U-Net.
        
        Args:
            x: Tenseur d'entrée [B, in_channels, H, W]
            
        Returns:
            Tenseur de sortie [B, out_channels, H, W]
        """
        # Stocker les caractéristiques d'encodeur pour les connexions de saut
        encoder_features = []
        
        # Passage à travers l'encodeur
        for i, encoder_block in enumerate(self.encoder_blocks):
            if i > 0:
                x = self.downsample_blocks[i-1](x)
            x = encoder_block(x)
            encoder_features.append(x)
        
        # Passage à travers le goulot d'étranglement
        x = self.bottleneck(x)
        
        # Passage à travers le décodeur
        for i in range(len(self.decoder_blocks)):
            # Récupérer les caractéristiques de l'encodeur
            skip = encoder_features[-(i+1)]
            
            # Appliquer l'attention sur les caractéristiques de l'encodeur
            # en utilisant les caractéristiques du décodeur comme signal de guidage
            skip_attention = self.attention_gates[i](g=x, x=skip)
            
            # Sur-échantillonnage et concaténation
            x = self.upsample_blocks[i](x)
            x = torch.cat([x, skip_attention], dim=1)
            
            # Convolution du décodeur
            x = self.decoder_blocks[i](x)
        
        # Convolution finale
        x = self.final_conv(x)
        
        # Activation finale si spécifiée
        if self.sigmoid is not None:
            x = self.sigmoid(x)
            
        return x
    
    def get_complexity(self) -> Dict[str, Any]:
        """
        Retourne des informations sur la complexité du modèle.
        
        Returns:
            Dictionnaire contenant des informations comme le nombre
            de paramètres, la taille mémoire, la profondeur, etc.
        """
        complexity = super().get_complexity()
        complexity.update({
            "encoder_blocks": len(self.encoder_blocks),
            "decoder_blocks": len(self.decoder_blocks),
            "attention_gates": len(self.attention_gates),
            "bottleneck_features": self.bottleneck.conv2.conv.out_channels
        })
        return complexity 