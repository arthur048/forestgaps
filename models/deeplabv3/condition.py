"""
Implémentation de DeepLabV3+ conditionnée par le seuil de hauteur.

Ce module fournit une implémentation PyTorch de l'architecture DeepLabV3+
qui intègre le seuil de hauteur comme condition pour la détection des trouées.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import ForestGapModel
from models.registry import ModelRegistry
from models.blocks.attention import CBAM
from models.deeplabv3.basic import ASPPModule, DecoderBlock


class PositionalEncoding(nn.Module):
    """
    Module d'encodage de position pour ajouter des informations spatiales.
    
    Ce module ajoute des informations de coordonnées normalisées
    aux features de l'encodeur.
    """
    
    def __init__(self):
        """
        Initialise le module d'encodage de position.
        """
        super(PositionalEncoding, self).__init__()
    
    def forward(self, x):
        """
        Ajoute des informations de position au tenseur d'entrée.
        
        Args:
            x: Tenseur d'entrée.
            
        Returns:
            Tenseur avec des canaux de position ajoutés.
        """
        batch_size, _, h, w = x.size()
        
        # Créer des grilles de coordonnées normalisées
        y_range = torch.linspace(0, 1, h, device=x.device)
        x_range = torch.linspace(0, 1, w, device=x.device)
        
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
        
        # Calculer la distance au centre
        y_center = torch.abs(y_grid - 0.5)
        x_center = torch.abs(x_grid - 0.5)
        
        # Réorganiser pour la concaténation
        y_grid = y_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        x_grid = x_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        y_center = y_center.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        x_center = x_center.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        
        # Concaténer avec les features existantes
        pos_encoding = torch.cat([y_grid, x_grid, y_center, x_center], dim=1)
        output = torch.cat([x, pos_encoding], dim=1)
        
        return output


class ThresholdConditioningModule(nn.Module):
    """
    Module de conditionnement par seuil pour DeepLabV3+.
    
    Ce module encode le seuil de hauteur et le projette dans l'espace
    des features pour moduler le comportement du réseau.
    """
    
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=128):
        """
        Initialise le module de conditionnement par seuil.
        
        Args:
            input_dim: Dimension d'entrée (généralement 1 pour un seuil scalaire).
            hidden_dim: Dimension cachée.
            output_dim: Dimension de sortie.
        """
        super(ThresholdConditioningModule, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # MLP pour encoder le seuil
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, threshold):
        """
        Encode le seuil pour le conditionnement.
        
        Args:
            threshold: Valeur de seuil [batch_size, 1].
            
        Returns:
            Features de conditionnement [batch_size, output_dim].
        """
        # Vérifier la dimension du seuil
        if threshold.dim() == 1:
            threshold = threshold.unsqueeze(1)
        
        # Encoder le seuil
        encoded = self.mlp(threshold)
        
        return encoded


@ModelRegistry.register("deeplabv3_plus_threshold")
class ThresholdConditionedDeepLabV3Plus(ForestGapModel):
    """
    Implémentation de DeepLabV3+ conditionnée par le seuil de hauteur.
    
    Ce modèle intègre le seuil de hauteur comme entrée supplémentaire pour
    adapter le comportement du réseau aux différents seuils de trouées.
    """
    
    def __init__(
        self, 
        in_channels=1, 
        out_channels=1, 
        encoder_channels=[64, 128, 256, 512],
        aspp_channels=256,
        decoder_channels=256,
        threshold_encoding_dim=128,
        dropout_rate=0.1,
        use_cbam=False,
        use_pos_encoding=True
    ):
        """
        Initialise le modèle DeepLabV3+ conditionné par seuil.
        
        Args:
            in_channels: Nombre de canaux d'entrée.
            out_channels: Nombre de canaux de sortie.
            encoder_channels: Liste des nombres de canaux pour chaque niveau de l'encodeur.
            aspp_channels: Nombre de canaux pour le module ASPP.
            decoder_channels: Nombre de canaux pour le décodeur.
            threshold_encoding_dim: Dimension de l'encodage du seuil.
            dropout_rate: Taux de dropout.
            use_cbam: Si True, utilise le module d'attention CBAM.
            use_pos_encoding: Si True, ajoute l'encodage de position aux features.
        """
        super(ThresholdConditionedDeepLabV3Plus, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channels = encoder_channels
        self.use_pos_encoding = use_pos_encoding
        
        # Ajustement des canaux d'entrée si on utilise l'encodage de position
        adjusted_in_channels = in_channels + 4 if use_pos_encoding else in_channels
        
        # Encodage de position
        self.pos_encoding = PositionalEncoding() if use_pos_encoding else None
        
        # Encodeur
        self.encoder_blocks = nn.ModuleList()
        in_ch = adjusted_in_channels
        
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
        
        # Module de conditionnement par seuil
        self.threshold_conditioning = ThresholdConditioningModule(
            input_dim=1,
            hidden_dim=64,
            output_dim=threshold_encoding_dim
        )
        
        # Intégration du seuil dans les features
        self.threshold_integration = nn.Sequential(
            nn.Conv2d(aspp_channels + threshold_encoding_dim, aspp_channels, 1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
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
    
    def forward(self, x, threshold):
        """
        Passe avant du modèle DeepLabV3+ conditionné par seuil.
        
        Args:
            x: Tenseur d'entrée [batch_size, in_channels, height, width].
            threshold: Seuil de hauteur [batch_size, 1].
            
        Returns:
            Tenseur de sortie [batch_size, out_channels, height, width].
        """
        # Encodage de position
        if self.use_pos_encoding:
            x = self.pos_encoding(x)
        
        # Stocker les features intermédiaires pour les skip connections
        features = []
        
        # Encodeur
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if i < len(self.encoder_blocks) - 1:
                features.append(x)
        
        # Module ASPP
        x = self.aspp(x)
        
        # Conditionnement par seuil
        threshold_encoding = self.threshold_conditioning(threshold)
        
        # Redimensionner l'encodage du seuil pour correspondre aux dimensions spatiales des features
        batch_size, _, h, w = x.size()
        threshold_encoding = threshold_encoding.view(batch_size, -1, 1, 1).expand(-1, -1, h, w)
        
        # Intégrer le seuil aux features
        x = torch.cat([x, threshold_encoding], dim=1)
        x = self.threshold_integration(x)
        
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
        return ["dsm", "threshold"]
    
    def get_output_names(self):
        """
        Retourne les noms des sorties du modèle.
        
        Returns:
            Liste des noms des sorties.
        """
        return ["mask"] 