"""
Implémentation de l'architecture Attention U-Net.

Ce module fournit une implémentation de l'architecture Attention U-Net qui
intègre des mécanismes d'attention dans l'architecture U-Net pour améliorer
les performances de segmentation.

Référence: Oktay, O., Schlemper, J., Folgoc, L. L., Lee, M., Heinrich, M., Misawa, K., ... & Glocker, B. (2018).
Attention U-Net: Learning Where to Look for the Pancreas. MIDL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union

from models.blocks.convolution import DoubleConvBlock
from models.blocks.pooling import DownsampleBlock, UpsampleBlock
from models.attention.cbam import CBAM


class AttentionGate(nn.Module):
    """
    Porte d'attention pour Attention U-Net.
    
    Cette classe implémente une porte d'attention qui permet au modèle
    de se concentrer sur les régions pertinentes de l'image.
    
    Attributes:
        W_g: Convolution pour les caractéristiques du décodeur
        W_x: Convolution pour les caractéristiques de l'encodeur
        psi: Convolution pour générer les poids d'attention
    """
    
    def __init__(self, g_channels: int, x_channels: int, int_channels: int):
        """
        Initialise la porte d'attention.
        
        Args:
            g_channels: Nombre de canaux des caractéristiques du décodeur
            x_channels: Nombre de canaux des caractéristiques de l'encodeur
            int_channels: Nombre de canaux intermédiaires
        """
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, int_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(int_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, int_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(int_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(int_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant de la porte d'attention.
        
        Args:
            g: Caractéristiques du décodeur [B, g_channels, H, W]
            x: Caractéristiques de l'encodeur [B, x_channels, H, W]
            
        Returns:
            Caractéristiques de l'encodeur pondérées par l'attention
        """
        # Adapter la taille de g si nécessaire
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
            
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class AttentionUNet(nn.Module):
    """
    Architecture Attention U-Net pour la segmentation d'images.
    
    Cette classe implémente une architecture U-Net avec des mécanismes d'attention
    pour améliorer les performances de segmentation.
    
    Attributes:
        encoder: Modules de l'encodeur
        bottleneck: Module du goulot d'étranglement
        attention_gates: Portes d'attention
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
        attention_type: str = 'gate',  # 'gate' ou 'cbam'
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU
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
            attention_type: Type de mécanisme d'attention ('gate' ou 'cbam')
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
        """
        super().__init__()
        
        self.depth = depth
        self.use_sigmoid = use_sigmoid
        self.attention_type = attention_type
        
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
        
        # Mécanismes d'attention
        if attention_type == 'gate':
            self.attention_gates = nn.ModuleList()
            for i in range(depth):
                g_channels = init_features * (2 ** (depth - i))
                x_channels = init_features * (2 ** (depth - i - 1))
                
                self.attention_gates.append(
                    AttentionGate(
                        g_channels=g_channels,
                        x_channels=x_channels,
                        int_channels=x_channels // 2
                    )
                )
        elif attention_type == 'cbam':
            self.attention_modules = nn.ModuleList()
            for i in range(depth):
                channels = init_features * (2 ** (depth - i - 1))
                
                self.attention_modules.append(
                    CBAM(channels=channels)
                )
        
        # Décodeur
        self.decoder = nn.ModuleList()
        
        # Niveaux du décodeur
        for i in range(depth):
            in_feats = init_features * (2 ** (depth - i))
            out_feats = init_features * (2 ** (depth - i - 1))
            skip_feats = init_features * (2 ** (depth - i - 1))
            
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
                        in_channels=out_feats + skip_feats,  # Caractéristiques upsampled + skip
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
        Passage avant du modèle Attention U-Net.
        
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
        
        # Décodeur avec connexions de saut et attention
        for i, decoder_block in enumerate(self.decoder):
            skip = encoder_outputs[-(i + 1)]
            
            # Upsampling
            x = decoder_block['upsample'](x)
            
            # Redimensionner si nécessaire
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            # Appliquer l'attention
            if self.attention_type == 'gate':
                skip = self.attention_gates[i](x, skip)
            elif self.attention_type == 'cbam' and i < len(self.attention_modules):
                skip = self.attention_modules[i](skip)
                
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