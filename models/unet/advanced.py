"""
Implémentation de l'architecture UNet3+ avancée.

Ce module fournit une implémentation de l'architecture UNet3+ qui
intègre des connexions denses entre les niveaux d'encodeur et de décodeur
pour améliorer les performances de segmentation.

Référence: Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., ... & Wu, J. (2020).
UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation. ICASSP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union

from models.blocks.convolution import ConvBlock


class UNet3Plus(nn.Module):
    """
    Architecture UNet3+ pour la segmentation d'images.
    
    Cette classe implémente l'architecture UNet3+ avec des connexions denses
    entre les niveaux d'encodeur et de décodeur pour améliorer les performances
    de segmentation.
    
    Attributes:
        encoder: Modules de l'encodeur
        decoder: Modules du décodeur
        final_conv: Convolution finale pour la prédiction
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 64,
        depth: int = 5,
        dropout_rate: float = 0.0,
        use_sigmoid: bool = True,
        deep_supervision: bool = False,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU
    ):
        """
        Initialise le modèle UNet3+.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie (classes)
            init_features: Nombre de caractéristiques initial
            depth: Profondeur du réseau (nombre de niveaux)
            dropout_rate: Taux de dropout
            use_sigmoid: Si True, applique une fonction sigmoid à la sortie
            deep_supervision: Si True, utilise la supervision profonde
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
        """
        super().__init__()
        
        self.depth = depth
        self.use_sigmoid = use_sigmoid
        self.deep_supervision = deep_supervision
        
        filters = [init_features * (2 ** i) for i in range(depth)]
        
        # Encodeur
        self.encoder = nn.ModuleList()
        
        # Premier niveau de l'encodeur (E1)
        self.encoder.append(
            nn.Sequential(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=filters[0],
                    kernel_size=3,
                    padding=1,
                    norm_layer=norm_layer,
                    activation=activation,
                    dropout_rate=dropout_rate
                ),
                ConvBlock(
                    in_channels=filters[0],
                    out_channels=filters[0],
                    kernel_size=3,
                    padding=1,
                    norm_layer=norm_layer,
                    activation=activation,
                    dropout_rate=dropout_rate
                )
            )
        )
        
        # Niveaux suivants de l'encodeur (E2-E5)
        for i in range(1, depth):
            self.encoder.append(
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    ConvBlock(
                        in_channels=filters[i-1],
                        out_channels=filters[i],
                        kernel_size=3,
                        padding=1,
                        norm_layer=norm_layer,
                        activation=activation,
                        dropout_rate=dropout_rate
                    ),
                    ConvBlock(
                        in_channels=filters[i],
                        out_channels=filters[i],
                        kernel_size=3,
                        padding=1,
                        norm_layer=norm_layer,
                        activation=activation,
                        dropout_rate=dropout_rate
                    )
                )
            )
        
        # Décodeur
        self.decoder = nn.ModuleList()
        
        # Taille des caractéristiques pour chaque niveau du décodeur
        self.decoder_filters = filters[0]  # Même nombre de filtres pour tous les niveaux du décodeur
        
        # Niveaux du décodeur (D1-D4)
        for i in range(depth - 1):
            # Dictionnaire pour stocker les convolutions pour chaque niveau d'encodeur
            level_convs = nn.ModuleDict()
            
            # Convolutions pour les caractéristiques de l'encodeur (E1-E5)
            for j in range(depth):
                # Calculer le facteur de redimensionnement
                if j < i:  # Niveau d'encodeur plus haut que le niveau de décodeur actuel
                    scale_factor = 2 ** (i - j)
                    level_convs[f'E{j+1}'] = nn.Sequential(
                        nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                        ConvBlock(
                            in_channels=filters[j],
                            out_channels=self.decoder_filters,
                            kernel_size=3,
                            padding=1,
                            norm_layer=norm_layer,
                            activation=activation,
                            dropout_rate=dropout_rate
                        )
                    )
                elif j > i:  # Niveau d'encodeur plus bas que le niveau de décodeur actuel
                    scale_factor = 2 ** (j - i)
                    level_convs[f'E{j+1}'] = nn.Sequential(
                        nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor),
                        ConvBlock(
                            in_channels=filters[j],
                            out_channels=self.decoder_filters,
                            kernel_size=3,
                            padding=1,
                            norm_layer=norm_layer,
                            activation=activation,
                            dropout_rate=dropout_rate
                        )
                    )
                else:  # Même niveau
                    level_convs[f'E{j+1}'] = ConvBlock(
                        in_channels=filters[j],
                        out_channels=self.decoder_filters,
                        kernel_size=3,
                        padding=1,
                        norm_layer=norm_layer,
                        activation=activation,
                        dropout_rate=dropout_rate
                    )
            
            # Convolutions pour les niveaux de décodeur précédents (D1-D3)
            for k in range(i):
                scale_factor = 2 ** (i - k)
                level_convs[f'D{k+1}'] = nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                    ConvBlock(
                        in_channels=self.decoder_filters,
                        out_channels=self.decoder_filters,
                        kernel_size=3,
                        padding=1,
                        norm_layer=norm_layer,
                        activation=activation,
                        dropout_rate=dropout_rate
                    )
                )
            
            # Convolution de fusion
            fusion = ConvBlock(
                in_channels=self.decoder_filters * (depth + i),  # Nombre total de sources
                out_channels=self.decoder_filters,
                kernel_size=3,
                padding=1,
                norm_layer=norm_layer,
                activation=activation,
                dropout_rate=dropout_rate
            )
            
            # Ajouter les convolutions et la fusion au décodeur
            self.decoder.append(
                nn.ModuleDict({
                    'convs': level_convs,
                    'fusion': fusion
                })
            )
        
        # Convolution finale
        self.final_conv = nn.Conv2d(
            in_channels=self.decoder_filters,
            out_channels=out_channels,
            kernel_size=1
        )
        
        # Supervision profonde (si activée)
        if deep_supervision:
            self.deep_supervision_convs = nn.ModuleList()
            for i in range(depth - 1):
                self.deep_supervision_convs.append(
                    nn.Conv2d(
                        in_channels=self.decoder_filters,
                        out_channels=out_channels,
                        kernel_size=1
                    )
                )
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Passage avant du modèle UNet3+.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur de sortie (segmentation) [B, out_channels, H, W]
            ou liste de tenseurs si deep_supervision=True
        """
        # Stocker les sorties de l'encodeur
        encoder_outputs = []
        
        # Encodeur
        for encoder_block in self.encoder:
            x = encoder_block(x)
            encoder_outputs.append(x)
        
        # Stocker les sorties du décodeur
        decoder_outputs = []
        
        # Décodeur
        for i, decoder_block in enumerate(self.decoder):
            # Collecter les caractéristiques de tous les niveaux d'encodeur
            features = []
            
            # Caractéristiques de l'encodeur
            for j in range(self.depth):
                features.append(decoder_block['convs'][f'E{j+1}'](encoder_outputs[j]))
            
            # Caractéristiques des niveaux de décodeur précédents
            for k in range(i):
                features.append(decoder_block['convs'][f'D{k+1}'](decoder_outputs[k]))
            
            # Concaténer toutes les caractéristiques
            x = torch.cat(features, dim=1)
            
            # Fusion
            x = decoder_block['fusion'](x)
            decoder_outputs.append(x)
        
        # Sortie finale
        if self.deep_supervision:
            outputs = []
            
            # Sorties de supervision profonde
            for i in range(len(decoder_outputs)):
                out = self.deep_supervision_convs[i](decoder_outputs[i])
                
                # Redimensionner à la taille d'entrée
                if out.shape[2:] != encoder_outputs[0].shape[2:]:
                    out = F.interpolate(
                        out, 
                        size=encoder_outputs[0].shape[2:], 
                        mode='bilinear', 
                        align_corners=True
                    )
                
                if self.use_sigmoid:
                    out = torch.sigmoid(out)
                    
                outputs.append(out)
            
            return outputs
        else:
            # Sortie principale
            x = self.final_conv(decoder_outputs[-1])
            
            # Redimensionner à la taille d'entrée si nécessaire
            if x.shape[2:] != encoder_outputs[0].shape[2:]:
                x = F.interpolate(
                    x, 
                    size=encoder_outputs[0].shape[2:], 
                    mode='bilinear', 
                    align_corners=True
                )
            
            if self.use_sigmoid:
                x = torch.sigmoid(x)
                
            return x 