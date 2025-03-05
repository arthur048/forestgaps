"""
Feature-wise Linear Modulation (FiLM) layers.

Ce module implémente les couches FiLM qui permettent de conditionner
les caractéristiques en fonction de paramètres externes.

Référence: Perez, E., Strub, F., De Vries, H., Dumoulin, V., & Courville, A. (2018).
FiLM: Visual Reasoning with a General Conditioning Layer. AAAI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """
    Couche Feature-wise Linear Modulation (FiLM).
    
    Cette couche applique une modulation linéaire aux caractéristiques
    en fonction de paramètres de conditionnement externes.
    
    Attributes:
        gamma: Paramètres de mise à l'échelle
        beta: Paramètres de décalage
    """
    
    def __init__(self, num_features: int):
        """
        Initialise la couche FiLM.
        
        Args:
            num_features: Nombre de canaux de caractéristiques
        """
        super().__init__()
        
        # Les paramètres gamma et beta seront fournis par un générateur externe
        self.num_features = num_features
        
    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Passage avant de la couche FiLM.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            gamma: Paramètres de mise à l'échelle [B, C]
            beta: Paramètres de décalage [B, C]
            
        Returns:
            Tenseur modulé
        """
        # Redimensionner gamma et beta pour la diffusion
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        
        # Appliquer la modulation linéaire
        return gamma * x + beta


class FiLMGenerator(nn.Module):
    """
    Générateur de paramètres FiLM.
    
    Ce module génère les paramètres gamma et beta pour la modulation FiLM
    en fonction d'un vecteur de conditionnement.
    
    Attributes:
        mlp: Réseau MLP pour générer les paramètres
    """
    
    def __init__(self, conditioning_size: int, num_features: int, hidden_dim: int = 128):
        """
        Initialise le générateur FiLM.
        
        Args:
            conditioning_size: Taille du vecteur de conditionnement
            num_features: Nombre de canaux de caractéristiques à moduler
            hidden_dim: Dimension cachée du MLP
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(conditioning_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_features * 2)  # gamma et beta
        )
        
        self.num_features = num_features
        
    def forward(self, conditioning: torch.Tensor) -> tuple:
        """
        Passage avant du générateur FiLM.
        
        Args:
            conditioning: Vecteur de conditionnement [B, conditioning_size]
            
        Returns:
            Tuple (gamma, beta) de paramètres pour la modulation FiLM
        """
        # Générer les paramètres gamma et beta
        params = self.mlp(conditioning)
        
        # Séparer gamma et beta
        gamma, beta = torch.split(params, self.num_features, dim=1)
        
        # Initialiser gamma autour de 1 pour une meilleure stabilité
        gamma = gamma + 1.0
        
        return gamma, beta


class AdaptiveFiLM(nn.Module):
    """
    Module FiLM adaptatif qui génère ses propres paramètres de conditionnement.
    
    Ce module combine un générateur FiLM et une couche FiLM, en générant
    les paramètres de conditionnement à partir des caractéristiques d'entrée.
    
    Attributes:
        generator: Générateur de paramètres FiLM
        film: Couche FiLM
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        """
        Initialise le module FiLM adaptatif.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            reduction_ratio: Ratio de réduction pour le générateur
        """
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.generator = FiLMGenerator(
            conditioning_size=in_channels,
            num_features=in_channels,
            hidden_dim=in_channels // reduction_ratio
        )
        self.film = FiLMLayer(num_features=in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du module FiLM adaptatif.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            
        Returns:
            Tenseur modulé
        """
        # Générer le vecteur de conditionnement à partir de l'entrée
        batch_size = x.size(0)
        conditioning = self.pool(x).view(batch_size, -1)
        
        # Générer les paramètres gamma et beta
        gamma, beta = self.generator(conditioning)
        
        # Appliquer la modulation FiLM
        return self.film(x, gamma, beta) 