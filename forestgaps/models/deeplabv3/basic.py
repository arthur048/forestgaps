"""
Implémentation de base de l'architecture DeepLabV3+ pour la détection des trouées forestières.

Ce module fournit une implémentation PyTorch de l'architecture DeepLabV3+,
connue pour ses performances dans les tâches de segmentation sémantique.
"""

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from forestgaps.models.base import ForestGapModel
from forestgaps.models.registry import ModelRegistry
from forestgaps.models.blocks.attention import CBAM


class ASPPModule(nn.Module):
    """
    Module ASPP (Atrous Spatial Pyramid Pooling) utilisé dans DeepLabV3+.
    
    Ce module utilise plusieurs convolutions dilatées en parallèle avec
    différents taux de dilatation pour capturer les caractéristiques
    multi-échelles.
    """
    
    def __init__(self, in_channels, out_channels, rates):
        """
        Initialise le module ASPP.
        
        Args:
            in_channels: Nombre de canaux d'entrée.
            out_channels: Nombre de canaux de sortie pour chaque branche.
            rates: Liste des taux de dilatation pour les convolutions dilatées.
        """
        super(ASPPModule, self).__init__()
        
        # 1x1 Convolution branch
        self.convs = nn.ModuleList()
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Dilated convolution branches
        for rate in rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Global pooling branch
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final 1x1
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        """
        Passe avant du module ASPP.
        
        Args:
            x: Tenseur d'entrée.
            
        Returns:
            Tenseur de sortie.
        """
        h, w = x.size(2), x.size(3)
        
        # Appliquer toutes les branches de convolution
        conv_outputs = [conv(x) for conv in self.convs]
        
        # Appliquer la branche globale
        global_features = self.global_branch(x)
        global_features = F.interpolate(global_features, size=(h, w), mode='bilinear', align_corners=True)
        
        # Concaténer toutes les branches
        concat_features = torch.cat(conv_outputs + [global_features], dim=1)
        
        # Convolution finale pour réduire le nombre de canaux
        output = self.final_conv(concat_features)
        
        return output


class DecoderBlock(nn.Module):
    """
    Bloc de décodeur pour DeepLabV3+.
    
    Ce bloc fusionne les features de l'encodeur avec les features du décodeur
    via des skip connections et des convolutions.
    """
    
    def __init__(self, low_level_channels, high_level_channels, out_channels, dropout_rate=0.1):
        """
        Initialise le bloc de décodeur.
        
        Args:
            low_level_channels: Nombre de canaux des features de bas niveau.
            high_level_channels: Nombre de canaux des features de haut niveau.
            out_channels: Nombre de canaux de sortie.
            dropout_rate: Taux de dropout.
        """
        super(DecoderBlock, self).__init__()
        
        # Réduction des canaux pour les features de bas niveau
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Convolution après concaténation
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(high_level_channels + 48, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, low_level_features, high_level_features):
        """
        Passe avant du bloc de décodeur.
        
        Args:
            low_level_features: Features de bas niveau de l'encodeur.
            high_level_features: Features de haut niveau du module ASPP.
            
        Returns:
            Features fusionnées.
        """
        # Réduire les canaux des features de bas niveau
        low_level_features = self.low_level_conv(low_level_features)
        
        # Redimensionner les features de haut niveau
        h, w = low_level_features.size(2), low_level_features.size(3)
        high_level_features = F.interpolate(high_level_features, size=(h, w), mode='bilinear', align_corners=True)
        
        # Concaténer les features
        combined_features = torch.cat([low_level_features, high_level_features], dim=1)
        
        # Appliquer les convolutions de fusion
        output = self.fusion_conv(combined_features)
        
        return output


@ModelRegistry.register("deeplabv3_plus")
class DeepLabV3Plus(ForestGapModel):
    """
    Implémentation de l'architecture DeepLabV3+ pour la détection des trouées forestières.
    
    Ce modèle combine un encodeur efficace avec le module ASPP et un décodeur
    simple pour la segmentation sémantique.
    """
    
    def __init__(
        self, 
        in_channels=1, 
        out_channels=1, 
        encoder_channels=[64, 128, 256, 512],
        aspp_channels=256,
        decoder_channels=256,
        dropout_rate=0.1,
        use_cbam=False
    ):
        """
        Initialise le modèle DeepLabV3+.
        
        Args:
            in_channels: Nombre de canaux d'entrée.
            out_channels: Nombre de canaux de sortie.
            encoder_channels: Liste des nombres de canaux pour chaque niveau de l'encodeur.
            aspp_channels: Nombre de canaux pour le module ASPP.
            decoder_channels: Nombre de canaux pour le décodeur.
            dropout_rate: Taux de dropout.
            use_cbam: Si True, utilise le module d'attention CBAM.
        """
        super(DeepLabV3Plus, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channels = encoder_channels
        
        # Encodeur
        self.encoder_blocks = nn.ModuleList()
        in_ch = in_channels
        
        for i, out_ch in enumerate(encoder_channels):
            block = []
            # Première couche avec stride=2 sauf pour le premier bloc
            stride = 2 if i > 0 else 1
            block.append(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False))
            block.append(nn.BatchNorm2d(out_ch))
            block.append(nn.ReLU(inplace=True))
            
            # Deuxième couche sans stride
            block.append(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False))
            block.append(nn.BatchNorm2d(out_ch))
            block.append(nn.ReLU(inplace=True))
            
            # Module CBAM optionnel
            if use_cbam:
                block.append(CBAM(out_ch))
            
            self.encoder_blocks.append(nn.Sequential(*block))
            in_ch = out_ch
        
        # Module ASPP
        self.aspp = ASPPModule(
            encoder_channels[-1], 
            aspp_channels, 
            rates=[6, 12, 18]
        )
        
        # Décodeur
        self.decoder = DecoderBlock(
            encoder_channels[0],
            aspp_channels,
            decoder_channels,
            dropout_rate
        )
        
        # Couche de sortie
        self.output_conv = nn.Conv2d(decoder_channels, out_channels, 1)
    
    def forward(self, x, threshold=None):
        """
        Passe avant du modèle DeepLabV3+.
        
        Args:
            x: Tenseur d'entrée.
            threshold: Seuil de hauteur (non utilisé dans cette version de base).
            
        Returns:
            Tenseur de sortie.
        """
        # Stocker les features intermédiaires pour les skip connections
        features = []
        
        # Encodeur
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if i < len(self.encoder_blocks) - 1:
                features.append(x)
        
        # Module ASPP
        x = self.aspp(x)
        
        # Décodeur
        x = self.decoder(features[0], x)
        
        # Upsampling final pour atteindre la taille d'entrée
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        
        # Couche de sortie
        x = self.output_conv(x)
        
        # Sigmoid pour obtenir une sortie entre 0 et 1
        x = torch.sigmoid(x)
        
        return x
        
    def get_input_names(self):
        """
        Retourne les noms des entrées du modèle.
        
        Returns:
            Liste des noms des entrées.
        """
        return ["dsm"]
        
    def get_output_names(self):
        """
        Retourne les noms des sorties du modèle.

        Returns:
            Liste des noms des sorties.
        """
        return ["mask"]

    def get_complexity(self) -> Dict[str, Any]:
        """
        Retourne des informations sur la complexité du modèle DeepLabV3+.

        Returns:
            Dictionnaire contenant des informations sur la complexité comme
            le nombre de paramètres, les canaux de l'encodeur, etc.
        """
        return {
            "parameters": self.get_num_parameters(),
            "encoder_channels": self.encoder_channels,
            "encoder_depth": len(self.encoder_channels),
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "model_type": "DeepLabV3Plus"
        } 