"""
Implémentation du modèle UNet avancé combinant FiLM, CBAM et DropPath.

Ce module fournit une implémentation de l'architecture UNet qui intègre
toutes les fonctionnalités avancées:
- FiLM pour le conditionnement par un seuil de hauteur
- CBAM pour l'attention aux caractéristiques pertinentes
- DropPath pour la régularisation des blocs profonds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Type, Union

from ..base import ThresholdConditionedUNet
from ..registry import model_registry
from ..blocks.residual_advanced import ResidualBlockWithFiLMCBAMDropPath
from ..blocks.downsampling import StridedConvDownsample
from ..blocks.upsampling import BilinearUpsampling


class FiLMGenerator(nn.Module):
    """
    Générateur de paramètres FiLM à partir de la condition (seuil de hauteur).
    
    Ce module transforme le seuil de hauteur (ou autre condition) en paramètres
    gamma et beta pour la modulation FiLM des caractéristiques à différents
    niveaux du réseau.
    """
    
    def __init__(
        self, 
        input_dim: int = 1, 
        hidden_dim: int = 64, 
        output_channels: List[int] = None
    ):
        """
        Initialise le générateur FiLM.
        
        Args:
            input_dim: Dimension d'entrée de la condition (généralement 1 pour le seuil)
            hidden_dim: Dimension cachée du réseau
            output_channels: Liste des nombres de canaux pour chaque niveau du réseau
        """
        super().__init__()
        
        if output_channels is None:
            raise ValueError("La liste des canaux de sortie ne peut pas être None")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        
        # Couche d'entrée pour encoder la condition
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Couches de sortie pour générer gamma et beta pour chaque niveau
        self.gamma_layers = nn.ModuleList()
        self.beta_layers = nn.ModuleList()
        
        for channels in output_channels:
            self.gamma_layers.append(nn.Linear(hidden_dim, channels))
            self.beta_layers.append(nn.Linear(hidden_dim, channels))
            
    def forward(self, condition: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Génère les paramètres FiLM à partir de la condition.
        
        Args:
            condition: Tensor de condition [B, input_dim]
            
        Returns:
            Liste de tuples (gamma, beta) pour chaque niveau du réseau
        """
        batch_size = condition.size(0)
        
        # Encoder la condition
        x = self.input_layer(condition)
        
        # Générer les paramètres pour chaque niveau
        film_params = []
        for gamma_layer, beta_layer in zip(self.gamma_layers, self.beta_layers):
            gamma = gamma_layer(x)
            beta = beta_layer(x)
            film_params.append((gamma, beta))
            
        return film_params


@model_registry.register("unet_all_features")
class UNetWithAllFeatures(ThresholdConditionedUNet):
    """
    UNet combinant FiLM, CBAM et DropPath pour la détection optimale des trouées forestières.
    
    Cette architecture avancée combine:
    - Conditionnement par seuil de hauteur via FiLM
    - Attention aux caractéristiques pertinentes via CBAM
    - Régularisation des couches profondes via DropPath
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        use_sigmoid: bool = True,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        reduction_ratio: int = 16
    ):
        """
        Initialise le modèle UNet avec toutes les fonctionnalités avancées.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie
            init_features: Nombre de caractéristiques initial
            depth: Profondeur du réseau (nombre de niveaux)
            dropout_rate: Taux de dropout standard
            drop_path_rate: Taux maximal de DropPath
            use_sigmoid: Appliquer sigmoid à la sortie pour normaliser entre 0 et 1
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
            reduction_ratio: Ratio de réduction pour les mécanismes d'attention
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
        
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.reduction_ratio = reduction_ratio
        
        # Liste des nombres de caractéristiques à chaque niveau
        features = [init_features * (2 ** i) for i in range(depth + 1)]
        
        # Générateur FiLM pour le conditionnement par seuil
        self.film_generator = FiLMGenerator(
            input_dim=1,  # Dimension du seuil
            hidden_dim=64,  # Dimension cachée du générateur
            output_channels=features  # Canaux à chaque niveau
        )
        
        # Encoder avec blocs résiduels avancés et DropPath
        self.encoder = nn.ModuleList()
        
        # Bloc d'entrée
        self.encoder.append(
            ResidualBlockWithFiLMCBAMDropPath(
                in_channels, 
                features[0],
                kernel_size=3,
                dropout_rate=dropout_rate,
                drop_path_rate=0.0,  # Pas de DropPath pour le premier bloc
                reduction_ratio=reduction_ratio,
                norm_layer=norm_layer,
                activation=activation
            )
        )
        
        # Blocs d'encodeur avec DropPath croissant avec la profondeur
        for i in range(1, depth + 1):
            # Augmenter progressivement le taux de DropPath avec la profondeur
            current_drop_path = drop_path_rate * (i / depth)
            
            self.encoder.append(
                nn.Sequential(
                    StridedConvDownsample(
                        features[i-1],
                        features[i],
                        scale_factor=2,
                        norm_layer=norm_layer,
                        activation=activation
                    ),
                    ResidualBlockWithFiLMCBAMDropPath(
                        features[i],
                        features[i],
                        kernel_size=3,
                        dropout_rate=dropout_rate,
                        drop_path_rate=current_drop_path,
                        reduction_ratio=reduction_ratio,
                        norm_layer=norm_layer,
                        activation=activation
                    )
                )
            )
        
        # Décodeur avec blocs résiduels avancés
        self.decoder = nn.ModuleList()
        for i in range(depth):
            self.decoder.append(
                nn.Sequential(
                    BilinearUpsampling(
                        features[depth-i],
                        features[depth-i-1],
                        scale_factor=2,
                        norm_layer=norm_layer,
                        activation=activation
                    ),
                    ResidualBlockWithFiLMCBAMDropPath(
                        features[depth-i-1] * 2,  # Concaténation avec skip
                        features[depth-i-1],
                        kernel_size=3,
                        dropout_rate=dropout_rate,
                        drop_path_rate=drop_path_rate * ((depth-i-1) / depth),  # DropPath décroissant
                        reduction_ratio=reduction_ratio,
                        norm_layer=norm_layer,
                        activation=activation
                    )
                )
            )
        
        # Couche de sortie
        self.final_conv = nn.Conv2d(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1
        )
        
    def forward(self, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """
        Propagation avant dans le UNet avec toutes les fonctionnalités.
        
        Args:
            x: Tensor d'entrée [B, C, H, W]
            threshold: Seuil de hauteur [B, 1]
            
        Returns:
            Tensor de sortie [B, out_channels, H, W]
        """
        # Générer les paramètres FiLM à partir du seuil
        film_params = self.film_generator(threshold)
        
        # Liste pour stocker les caractéristiques de skip connections
        skip_connections = []
        
        # Encoder
        out = x
        for i, enc_block in enumerate(self.encoder):
            if i == 0:
                # Premier bloc avec FiLM
                gamma, beta = film_params[i]
                out = enc_block(out, gamma, beta)
            else:
                # Blocs suivants (downsample + résiduel avec FiLM)
                out = enc_block[0](out)  # Downsampling
                gamma, beta = film_params[i]
                out = enc_block[1](out, gamma, beta)  # Bloc résiduel with FiLM
                
            # Sauvegarder pour skip connection (sauf le dernier niveau)
            if i < self.depth:
                skip_connections.append(out)
        
        # Décodeur avec skip connections
        for i, dec_block in enumerate(self.decoder):
            # Upsampling
            out = dec_block[0](out)
            
            # Récupérer la skip connection
            skip = skip_connections[self.depth - i - 1]
            
            # Concaténer
            out = torch.cat([out, skip], dim=1)
            
            # Appliquer le bloc résiduel avec FiLM
            gamma, beta = film_params[self.depth - i - 1]
            out = dec_block[1](out, gamma, beta)
        
        # Couche finale
        out = self.final_conv(out)
        
        # Appliquer sigmoid si demandé
        if self.use_sigmoid:
            out = torch.sigmoid(out)
            
        return out
    
    def get_complexity(self) -> Dict[str, Any]:
        """
        Calcule la complexité du modèle.
        
        Returns:
            Dict contenant les informations de complexité
        """
        # Calculer le nombre de paramètres
        num_params = self.get_num_parameters()
        
        # Calculer la taille de l'architecture
        features = [(2 ** i) * self.init_features for i in range(self.depth + 1)]
        architecture_info = {
            "type": "UNetWithAllFeatures",
            "depth": self.depth,
            "init_features": self.init_features,
            "features_per_level": features,
            "drop_path_rate": self.drop_path_rate,
            "reduction_ratio": self.reduction_ratio,
            "dropout_rate": self.dropout_rate
        }
        
        return {
            "num_parameters": num_params,
            "architecture": architecture_info
        } 