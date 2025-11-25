"""
Implémentation de l'architecture FiLM U-Net pour la segmentation d'images forestières.

Ce module fournit une implémentation de l'architecture FiLM U-Net qui intègre
des mécanismes de modulation de caractéristiques (Feature-wise Linear Modulation)
pour conditionner le modèle sur des paramètres externes comme le seuil de hauteur.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Type, Union

from ..base import ThresholdConditionedUNet
from ..registry import model_registry
from ..blocks.conv import DoubleConvBlock
from ..blocks.downsampling import MaxPoolDownsample
from ..blocks.upsampling import BilinearUpsampling


class FiLMLayer(nn.Module):
    """
    Couche de modulation de caractéristiques (Feature-wise Linear Modulation).
    
    Cette couche applique une transformation affine aux caractéristiques
    en fonction d'un paramètre de conditionnement externe.
    """
    
    def __init__(
        self,
        feature_channels: int,
        condition_size: int,
        hidden_size: Optional[int] = None
    ):
        """
        Initialise une couche FiLM.
        
        Args:
            feature_channels: Nombre de canaux des caractéristiques à moduler
            condition_size: Taille du vecteur de conditionnement
            hidden_size: Taille de la couche cachée pour le réseau de modulation
        """
        super().__init__()
        
        if hidden_size is None:
            hidden_size = max(condition_size, feature_channels)
            
        # Réseau pour générer les paramètres gamma (scale) et beta (shift)
        self.film_generator = nn.Sequential(
            nn.Linear(condition_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, feature_channels * 2)  # gamma et beta
        )
        
        # Initialisation des poids pour que gamma soit proche de 1 et beta de 0
        self.film_generator[2].weight.data.zero_()
        self.film_generator[2].bias.data.zero_()
        
        self.feature_channels = feature_channels
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Passage avant de la couche FiLM.
        
        Args:
            x: Tenseur des caractéristiques à moduler [B, C, H, W]
            condition: Tenseur de conditionnement [B, condition_size]
            
        Returns:
            Tenseur des caractéristiques modulées [B, C, H, W]
        """
        batch_size = x.size(0)
        
        # Générer les paramètres gamma et beta
        film_params = self.film_generator(condition)
        
        # Séparer gamma et beta
        gamma, beta = torch.split(film_params, self.feature_channels, dim=1)
        
        # Redimensionner pour la diffusion
        gamma = gamma.view(batch_size, self.feature_channels, 1, 1)
        beta = beta.view(batch_size, self.feature_channels, 1, 1)
        
        # Appliquer la modulation: gamma * x + beta
        return gamma * x + beta


class FiLMDoubleConvBlock(nn.Module):
    """
    Bloc à double convolution avec modulation FiLM.
    
    Ce bloc étend le bloc à double convolution standard en ajoutant
    une modulation FiLM après chaque convolution.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_size: int,
        mid_channels: Optional[int] = None,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        dropout_rate: float = 0.0
    ):
        """
        Initialise un bloc à double convolution avec FiLM.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            condition_size: Taille du vecteur de conditionnement
            mid_channels: Nombre de canaux intermédiaires
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
            dropout_rate: Taux de dropout
        """
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
            
        # Première convolution
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.norm1 = norm_layer(mid_channels)
        self.film1 = FiLMLayer(mid_channels, condition_size)
        self.activation1 = activation(inplace=True)
        
        # Seconde convolution
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.norm2 = norm_layer(out_channels)
        self.film2 = FiLMLayer(out_channels, condition_size)
        self.activation2 = activation(inplace=True)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du bloc à double convolution avec FiLM.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            condition: Tenseur de conditionnement [B, condition_size]
            
        Returns:
            Tenseur de sortie après les convolutions et modulations
        """
        # Première convolution avec FiLM
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.film1(x, condition)
        x = self.activation1(x)
        
        # Seconde convolution avec FiLM
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film2(x, condition)
        x = self.activation2(x)
        
        # Dropout si spécifié
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x


@model_registry.register("film_unet")
class FiLMUNet(ThresholdConditionedUNet):
    """
    Implémentation de l'architecture FiLM U-Net.
    
    Cette classe implémente l'architecture FiLM U-Net qui utilise des
    mécanismes de modulation de caractéristiques pour conditionner le modèle
    sur des paramètres externes comme le seuil de hauteur pour la détection
    des trouées forestières.
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
        activation: Type[nn.Module] = nn.ReLU,
        condition_size: int = 1  # Taille du vecteur de conditionnement (seuil)
    ):
        """
        Initialise le modèle FiLM U-Net.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie (classes)
            init_features: Nombre de caractéristiques initial
            depth: Profondeur du réseau (nombre de niveaux)
            dropout_rate: Taux de dropout
            use_sigmoid: Si True, applique une fonction sigmoid à la sortie
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
            condition_size: Taille du vecteur de conditionnement
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
        
        self.condition_size = condition_size
        
        # Couche d'embedding pour le seuil
        self.threshold_embedding = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, condition_size)
        )
        
        # Initialiser les listes pour stocker les couches
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        # Construire l'encodeur
        features = init_features
        encoder_features_channels = [features]
        
        # Premier bloc d'encodeur (pas de downsampling)
        self.encoder_blocks.append(
            FiLMDoubleConvBlock(
                in_channels=in_channels,
                out_channels=features,
                condition_size=condition_size,
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
                FiLMDoubleConvBlock(
                    in_channels=in_features,
                    out_channels=features,
                    condition_size=condition_size,
                    norm_layer=norm_layer,
                    activation=activation,
                    dropout_rate=dropout_rate if i == depth - 2 else 0.0
                )
            )
            
            encoder_features_channels.append(features)
        
        # Goulot d'étranglement (bottleneck)
        self.bottleneck = FiLMDoubleConvBlock(
            in_channels=features,
            out_channels=features * 2,
            condition_size=condition_size,
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
                FiLMDoubleConvBlock(
                    in_channels=out_features * 2,  # Concaténation avec skip
                    out_channels=out_features,
                    condition_size=condition_size,
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
        
    def forward(self, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du modèle FiLM U-Net.
        
        Args:
            x: Tenseur d'entrée [B, in_channels, H, W]
            threshold: Tenseur des seuils de hauteur [B, 1]
            
        Returns:
            Tenseur de sortie [B, out_channels, H, W]
        """
        # Encoder le seuil
        condition = self.threshold_embedding(threshold)
        
        # Stocker les caractéristiques d'encodeur pour les connexions de saut
        encoder_features = []
        
        # Passage à travers l'encodeur
        for i, encoder_block in enumerate(self.encoder_blocks):
            if i > 0:
                x = self.downsample_blocks[i-1](x)
            x = encoder_block(x, condition)
            encoder_features.append(x)
        
        # Passage à travers le goulot d'étranglement
        x = self.bottleneck(x, condition)
        
        # Passage à travers le décodeur
        for i in range(len(self.decoder_blocks)):
            # Récupérer les caractéristiques de l'encodeur
            skip = encoder_features[-(i+1)]
            
            # Sur-échantillonnage et concaténation
            x = self.upsample_blocks[i](x, skip)
            
            # Convolution du décodeur avec FiLM
            x = self.decoder_blocks[i](x, condition)
        
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
            "condition_size": self.condition_size,
            "model_type": "FiLMUNet"
        })
        return complexity 