#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemple de création d'un modèle personnalisé pour ForestGaps.

Ce script montre comment créer, enregistrer et utiliser un modèle
personnalisé en exploitant le système de registre des modèles.

Auteur: Arthur VDL
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Assurer que le package est dans le PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forestgaps.models.registry import ModelRegistry, register_model
from forestgaps.models.blocks import ResidualBlock
from forestgaps.environment import setup_environment

# ===================================================================================================
# CRÉATION D'UN MODÈLE PERSONNALISÉ
# ===================================================================================================

class DeepResUNet(nn.Module):
    """
    U-Net avec des blocs résiduels profonds et canal de condition pour le seuil.
    
    Ce modèle personnalisé utilise une architecture U-Net avec des blocs résiduels
    plus profonds et un mécanisme de modulation de caractéristiques basé sur le seuil.
    """
    
    def __init__(self, in_channels=1, depth=5, initial_features=32, dropout_rate=0.2):
        """
        Initialiser le modèle DeepResUNet.
        
        Args:
            in_channels (int): Nombre de canaux d'entrée (défaut: 1).
            depth (int): Profondeur du réseau U-Net (défaut: 5).
            initial_features (int): Nombre de caractéristiques initiales (défaut: 32).
            dropout_rate (float): Taux de dropout (défaut: 0.2).
        """
        super().__init__()
        
        self.depth = depth
        
        # Module de traitement du seuil (condition)
        self.threshold_processor = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # Encodeur
        self.encoder_blocks = nn.ModuleList()
        
        # Premier bloc (pas de downsampling)
        self.encoder_blocks.append(
            nn.Sequential(
                ResidualBlock(in_channels, initial_features, dropout_rate=dropout_rate),
                ResidualBlock(initial_features, initial_features, dropout_rate=dropout_rate)
            )
        )
        
        # Blocs d'encodeur restants
        for i in range(1, depth):
            in_features = initial_features * (2**(i-1))
            out_features = initial_features * (2**i)
            
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    ResidualBlock(in_features, out_features, dropout_rate=dropout_rate),
                    ResidualBlock(out_features, out_features, dropout_rate=dropout_rate)
                )
            )
        
        # Décodeur
        self.decoder_blocks = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        
        # Blocs de décodeur
        for i in range(depth-1, 0, -1):
            in_features = initial_features * (2**i)
            out_features = initial_features * (2**(i-1))
            
            self.upsampling_layers.append(
                nn.ConvTranspose2d(in_features, out_features, kernel_size=2, stride=2)
            )
            
            self.decoder_blocks.append(
                nn.Sequential(
                    ResidualBlock(in_features, out_features, dropout_rate=dropout_rate),
                    ResidualBlock(out_features, out_features, dropout_rate=dropout_rate)
                )
            )
        
        # Couche de sortie (segmentation)
        self.output_conv = nn.Conv2d(initial_features, 1, kernel_size=1)
    
    def encode(self, x):
        """
        Encoder l'entrée.
        
        Args:
            x (torch.Tensor): Tenseur d'entrée [B, C, H, W].
            
        Returns:
            list: Liste des activations d'encodeur.
        """
        features = []
        
        # Passer à travers les blocs d'encodeur
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
        
        return features
    
    def decode(self, features, condition_features):
        """
        Décoder les caractéristiques.
        
        Args:
            features (list): Liste des activations d'encodeur.
            condition_features (torch.Tensor): Caractéristiques de condition.
            
        Returns:
            torch.Tensor: Sorties du décodeur.
        """
        # Commencer avec les caractéristiques les plus profondes
        x = features[-1]
        
        # Parcourir les blocs de décodeur
        for i in range(len(self.decoder_blocks)):
            # Upsampling
            x = self.upsampling_layers[i](x)
            
            # Caractéristiques de skip-connection
            skip_features = features[-(i+2)]
            
            # Concaténer les caractéristiques
            x = torch.cat([x, skip_features], dim=1)
            
            # Appliquer le bloc de décodeur
            x = self.decoder_blocks[i](x)
            
            # Appliquer la modulation de condition
            # (simple addition des caractéristiques de condition adaptées à la forme)
            condition_adapted = condition_features.unsqueeze(-1).unsqueeze(-1)
            condition_adapted = condition_adapted.expand(-1, -1, x.size(2), x.size(3))
            
            # Adapter les dimensions des caractéristiques de condition si nécessaire
            if condition_adapted.size(1) < x.size(1):
                condition_adapted = F.pad(condition_adapted, (0, 0, 0, 0, 0, x.size(1) - condition_adapted.size(1)))
            elif condition_adapted.size(1) > x.size(1):
                condition_adapted = condition_adapted[:, :x.size(1)]
            
            # Ajouter la condition modulation
            x = x + 0.1 * condition_adapted
        
        return x
    
    def forward(self, x, threshold):
        """
        Passe avant du modèle.
        
        Args:
            x (torch.Tensor): Tenseur d'entrée [B, C, H, W].
            threshold (torch.Tensor): Valeur de seuil [B].
            
        Returns:
            torch.Tensor: Prédictions de segmentation [B, 1, H, W].
        """
        # Traiter la condition (seuil)
        threshold = threshold.view(-1, 1)  # [B, 1]
        condition_features = self.threshold_processor(threshold)  # [B, 32]
        
        # Encoder
        features = self.encode(x)
        
        # Decoder
        decoded = self.decode(features, condition_features)
        
        # Couche de sortie
        output = self.output_conv(decoded)
        
        return output

# ===================================================================================================
# ENREGISTREMENT DU MODÈLE PERSONNALISÉ
# ===================================================================================================

# Méthode 1: Enregistrement explicite
ModelRegistry.register("deep_resunet", DeepResUNet)

# Méthode 2: Enregistrement via décorateur
@register_model("deep_resunet_v2")
class DeepResUNetV2(DeepResUNet):
    """Version améliorée du DeepResUNet avec des fonctionnalités supplémentaires."""
    
    def __init__(self, in_channels=1, depth=5, initial_features=32, dropout_rate=0.2, use_attention=True):
        """
        Initialiser le modèle DeepResUNetV2.
        
        Args:
            in_channels (int): Nombre de canaux d'entrée (défaut: 1).
            depth (int): Profondeur du réseau U-Net (défaut: 5).
            initial_features (int): Nombre de caractéristiques initiales (défaut: 32).
            dropout_rate (float): Taux de dropout (défaut: 0.2).
            use_attention (bool): Utiliser des mécanismes d'attention (défaut: True).
        """
        super().__init__(in_channels, depth, initial_features, dropout_rate)
        self.use_attention = use_attention
        
        # Ajouter des couches d'attention si demandé
        if use_attention:
            # Créer des blocs d'attention pour le décodeur
            self.attention_blocks = nn.ModuleList()
            
            for i in range(depth-1, 0, -1):
                out_features = initial_features * (2**(i-1))
                
                # Bloc d'attention simple
                self.attention_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(out_features, out_features, kernel_size=1),
                        nn.Sigmoid()
                    )
                )
    
    def decode(self, features, condition_features):
        """
        Décoder les caractéristiques avec attention.
        
        Args:
            features (list): Liste des activations d'encodeur.
            condition_features (torch.Tensor): Caractéristiques de condition.
            
        Returns:
            torch.Tensor: Sorties du décodeur.
        """
        # Commencer avec les caractéristiques les plus profondes
        x = features[-1]
        
        # Parcourir les blocs de décodeur
        for i in range(len(self.decoder_blocks)):
            # Upsampling
            x = self.upsampling_layers[i](x)
            
            # Caractéristiques de skip-connection
            skip_features = features[-(i+2)]
            
            # Concaténer les caractéristiques
            x = torch.cat([x, skip_features], dim=1)
            
            # Appliquer le bloc de décodeur
            x = self.decoder_blocks[i](x)
            
            # Appliquer l'attention si activée
            if self.use_attention:
                attention_mask = self.attention_blocks[i](x)
                x = x * attention_mask
            
            # Appliquer la modulation de condition comme dans la classe de base
            condition_adapted = condition_features.unsqueeze(-1).unsqueeze(-1)
            condition_adapted = condition_adapted.expand(-1, -1, x.size(2), x.size(3))
            
            if condition_adapted.size(1) < x.size(1):
                condition_adapted = F.pad(condition_adapted, (0, 0, 0, 0, 0, x.size(1) - condition_adapted.size(1)))
            elif condition_adapted.size(1) > x.size(1):
                condition_adapted = condition_adapted[:, :x.size(1)]
            
            x = x + 0.1 * condition_adapted
        
        return x

# ===================================================================================================
# DÉMONSTRATION DU MODÈLE PERSONNALISÉ
# ===================================================================================================

def test_custom_model():
    """
    Tester le modèle personnalisé avec des données synthétiques.
    """
    # Configurer le logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Configurer l'environnement
    env = setup_environment()
    device = env.get_device()
    logger.info(f"Environnement: {env.name}, Dispositif: {device}")
    
    # Afficher les modèles disponibles
    available_models = ModelRegistry.list_available_models()
    logger.info(f"Modèles disponibles: {', '.join(available_models)}")
    
    # Créer une instance du modèle personnalisé
    logger.info("Création du modèle personnalisé...")
    model = ModelRegistry.create(
        "deep_resunet",
        in_channels=1,
        depth=4,
        initial_features=32,
        dropout_rate=0.2
    )
    model.to(device)
    
    # Créer des données synthétiques
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 256, 256, device=device)
    threshold = torch.tensor([5.0, 10.0], device=device)  # Différents seuils pour chaque exemple
    
    # Passer en mode évaluation et exécuter le modèle
    model.eval()
    logger.info("Exécution de l'inférence...")
    
    with torch.no_grad():
        output = model(input_tensor, threshold)
    
    # Vérifier la forme de sortie
    logger.info(f"Forme d'entrée: {input_tensor.shape}")
    logger.info(f"Forme de sortie: {output.shape}")
    
    # Tester la version 2 du modèle
    logger.info("Création du modèle personnalisé V2...")
    model_v2 = ModelRegistry.create(
        "deep_resunet_v2",
        in_channels=1,
        depth=4,
        initial_features=32,
        dropout_rate=0.2,
        use_attention=True
    )
    model_v2.to(device)
    
    # Passer en mode évaluation et exécuter le modèle V2
    model_v2.eval()
    logger.info("Exécution de l'inférence avec le modèle V2...")
    
    with torch.no_grad():
        output_v2 = model_v2(input_tensor, threshold)
    
    logger.info(f"Forme de sortie V2: {output_v2.shape}")
    logger.info("Test des modèles personnalisés terminé.")
    
    return {
        "model": model,
        "model_v2": model_v2,
        "input": input_tensor,
        "threshold": threshold,
        "output": output,
        "output_v2": output_v2
    }

# ===================================================================================================
# POINT D'ENTRÉE
# ===================================================================================================

if __name__ == "__main__":
    results = test_custom_model()
    
    print("\nExemple de création et d'utilisation de modèles personnalisés terminé avec succès.")
    print(f"Modèles disponibles dans le registre: {', '.join(ModelRegistry.list_available_models())}")
    
    # Afficher les caractéristiques principales des modèles
    for model_name, model in [("DeepResUNet", results["model"]), ("DeepResUNetV2", results["model_v2"])]:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModèle {model_name}:")
        print(f"  Paramètres totaux: {total_params:,}")
        print(f"  Paramètres entraînables: {trainable_params:,}")
        print(f"  Taille d'entrée: {results['input'].shape}")
        print(f"  Taille de sortie: {results['output'].shape}") 