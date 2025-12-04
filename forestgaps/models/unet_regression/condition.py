"""
Implémentation de U-Net conditionné par seuil pour la régression.

Ce module fournit une implémentation PyTorch de l'architecture U-Net
conditionnée par seuil pour les tâches de régression sur des données forestières.
"""

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from forestgaps.models.base import ForestGapModel
from forestgaps.models.registry import ModelRegistry
from forestgaps.models.unet_regression.basic import DoubleConv, Down, Up, OutConv


class ThresholdEncoder(nn.Module):
    """
    Module d'encodage du seuil pour conditionner le réseau.
    
    Ce module encode le seuil de hauteur en un vecteur de features
    qui pourra être utilisé pour conditionner le comportement du réseau.
    """
    
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=64):
        """
        Initialise l'encodeur de seuil.
        
        Args:
            input_dim: Dimension d'entrée (généralement 1 pour un seuil scalaire).
            hidden_dim: Dimension cachée.
            output_dim: Dimension de sortie.
        """
        super(ThresholdEncoder, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, threshold):
        """
        Encode le seuil en un vecteur de features.
        
        Args:
            threshold: Tenseur de seuil [batch_size, 1].
            
        Returns:
            Tenseur encodé [batch_size, output_dim].
        """
        # S'assurer que threshold a la bonne forme
        if threshold.dim() == 1:
            threshold = threshold.unsqueeze(1)
            
        return self.mlp(threshold)


class FiLMLayer(nn.Module):
    """
    Couche FiLM (Feature-wise Linear Modulation) pour conditionner les features.
    
    Cette couche applique une modulation linéaire des features
    basée sur une condition (le seuil encodé).
    """
    
    def __init__(self, feature_dim, condition_dim):
        """
        Initialise la couche FiLM.
        
        Args:
            feature_dim: Dimension des features à moduler.
            condition_dim: Dimension de la condition (seuil encodé).
        """
        super(FiLMLayer, self).__init__()
        
        # Projections pour générer gamma et beta
        self.gamma_projection = nn.Linear(condition_dim, feature_dim)
        self.beta_projection = nn.Linear(condition_dim, feature_dim)
    
    def forward(self, features, condition):
        """
        Applique la modulation FiLM aux features.
        
        Args:
            features: Tenseur de features [batch_size, feature_dim, height, width].
            condition: Tenseur de condition [batch_size, condition_dim].
            
        Returns:
            Tenseur modulé [batch_size, feature_dim, height, width].
        """
        batch_size, feature_dim, height, width = features.size()
        
        # Calculer gamma et beta
        gamma = self.gamma_projection(condition)  # [batch_size, feature_dim]
        beta = self.beta_projection(condition)    # [batch_size, feature_dim]
        
        # Redimensionner pour la diffusion
        gamma = gamma.view(batch_size, feature_dim, 1, 1)
        beta = beta.view(batch_size, feature_dim, 1, 1)
        
        # Appliquer la modulation
        return features * (1 + gamma) + beta


@ModelRegistry.register("regression_unet_threshold")
class ThresholdConditionedRegressionUNet(ForestGapModel):
    """
    Implémentation de U-Net pour régression conditionnée par seuil.
    
    Cette architecture est adaptée pour prédire des valeurs continues
    (régression) en utilisant le seuil de hauteur comme condition.
    """
    
    def __init__(
        self, 
        in_channels=1, 
        out_channels=1, 
        init_features=64,
        condition_dim=64,
        bilinear=True,
        dropout_rate=0.2
    ):
        """
        Initialise le modèle U-Net pour régression conditionnée par seuil.
        
        Args:
            in_channels: Nombre de canaux d'entrée.
            out_channels: Nombre de canaux de sortie.
            init_features: Nombre de features dans la première couche.
            condition_dim: Dimension de l'encodage du seuil.
            bilinear: Si True, utilise l'interpolation bilinéaire pour l'upsampling.
            dropout_rate: Taux de dropout à appliquer.
        """
        super(ThresholdConditionedRegressionUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.init_features = init_features
        self.condition_dim = condition_dim
        
        # Encodeur de seuil
        self.threshold_encoder = ThresholdEncoder(
            input_dim=1,
            hidden_dim=64,
            output_dim=condition_dim
        )
        
        # Encodeur
        self.inc = DoubleConv(in_channels, init_features, dropout_rate=dropout_rate)
        self.down1 = Down(init_features, init_features * 2, dropout_rate=dropout_rate)
        self.down2 = Down(init_features * 2, init_features * 4, dropout_rate=dropout_rate)
        self.down3 = Down(init_features * 4, init_features * 8, dropout_rate=dropout_rate)
        factor = 2 if bilinear else 1
        self.down4 = Down(init_features * 8, init_features * 16 // factor, dropout_rate=dropout_rate)
        
        # Couches FiLM pour conditionner les features
        self.film1 = FiLMLayer(init_features, condition_dim)
        self.film2 = FiLMLayer(init_features * 2, condition_dim)
        self.film3 = FiLMLayer(init_features * 4, condition_dim)
        self.film4 = FiLMLayer(init_features * 8, condition_dim)
        self.film5 = FiLMLayer(init_features * 16 // factor, condition_dim)
        
        # Décodeur
        self.up1 = Up(init_features * 16, init_features * 8 // factor, bilinear, dropout_rate=dropout_rate)
        self.up2 = Up(init_features * 8, init_features * 4 // factor, bilinear, dropout_rate=dropout_rate)
        self.up3 = Up(init_features * 4, init_features * 2 // factor, bilinear, dropout_rate=dropout_rate)
        self.up4 = Up(init_features * 2, init_features, bilinear, dropout_rate=dropout_rate)
        self.outc = OutConv(init_features, out_channels)
    
    def forward(self, x, threshold):
        """
        Passe avant du modèle U-Net pour régression conditionnée par seuil.
        
        Args:
            x: Tenseur d'entrée [batch_size, in_channels, height, width].
            threshold: Tenseur de seuil [batch_size, 1].
            
        Returns:
            Tenseur de sortie [batch_size, out_channels, height, width].
        """
        # Encoder le seuil
        threshold_encoding = self.threshold_encoder(threshold)
        
        # Encodeur avec conditionnement FiLM
        x1 = self.inc(x)
        x1 = self.film1(x1, threshold_encoding)
        
        x2 = self.down1(x1)
        x2 = self.film2(x2, threshold_encoding)
        
        x3 = self.down2(x2)
        x3 = self.film3(x3, threshold_encoding)
        
        x4 = self.down3(x3)
        x4 = self.film4(x4, threshold_encoding)
        
        x5 = self.down4(x4)
        x5 = self.film5(x5, threshold_encoding)
        
        # Décodeur
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        # Pour la régression, pas de sigmoid à la fin
        return x
    
    def get_input_names(self):
        """
        Retourne les noms des entrées du modèle.
        
        Returns:
            Liste des noms des entrées.
        """
        return ["dsm", "threshold"]
    
    def get_output_names(self):
        """
        Retourne les noms des sorties du modèle.

        Returns:
            Liste des noms des sorties.
        """
        return ["prediction"]

    def get_complexity(self) -> Dict[str, Any]:
        """
        Retourne des informations sur la complexité du modèle.

        Returns:
            Dictionnaire contenant des informations sur la complexité.
        """
        return {
            "parameters": self.get_num_parameters(),
            "init_features": self.init_features,
            "condition_dim": self.condition_dim,
            "depth": 4,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "bilinear": self.bilinear,
            "model_type": "ThresholdConditionedRegressionUNet"
        } 