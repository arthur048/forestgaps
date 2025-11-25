"""
Implémentation de base de l'architecture U-Net pour les tâches de régression.

Ce module fournit une implémentation PyTorch de l'architecture U-Net
adaptée pour prédire des valeurs continues (régression) sur des données forestières.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from forestgaps.models.base import ForestGapModel
from forestgaps.models.registry import ModelRegistry


class DoubleConv(nn.Module):
    """
    Bloc de double convolution utilisé dans U-Net.
    
    Ce bloc applique deux convolutions 3x3 consécutives, chacune suivie
    d'une normalisation par batch et d'une activation ReLU.
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.0):
        """
        Initialise le bloc de double convolution.
        
        Args:
            in_channels: Nombre de canaux d'entrée.
            out_channels: Nombre de canaux de sortie.
            mid_channels: Nombre de canaux intermédiaires (si None, utilise out_channels).
            dropout_rate: Taux de dropout à appliquer entre les convolutions.
        """
        super(DoubleConv, self).__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Passe avant du bloc de double convolution.
        
        Args:
            x: Tenseur d'entrée.
            
        Returns:
            Tenseur de sortie après les deux convolutions.
        """
        return self.double_conv(x)


class Down(nn.Module):
    """
    Bloc de down-sampling utilisé dans U-Net.
    
    Ce bloc applique un max pooling 2x2 suivi d'un bloc de double convolution.
    """
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        """
        Initialise le bloc de down-sampling.
        
        Args:
            in_channels: Nombre de canaux d'entrée.
            out_channels: Nombre de canaux de sortie.
            dropout_rate: Taux de dropout à appliquer dans la double convolution.
        """
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        )
    
    def forward(self, x):
        """
        Passe avant du bloc de down-sampling.
        
        Args:
            x: Tenseur d'entrée.
            
        Returns:
            Tenseur de sortie après pooling et convolutions.
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Bloc de up-sampling utilisé dans U-Net.
    
    Ce bloc applique un up-sampling (transposed conv ou interpolation)
    suivi d'un bloc de double convolution.
    """
    
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.0):
        """
        Initialise le bloc de up-sampling.
        
        Args:
            in_channels: Nombre de canaux d'entrée.
            out_channels: Nombre de canaux de sortie.
            bilinear: Si True, utilise l'interpolation bilinéaire,
                      sinon utilise la convolution transposée.
            dropout_rate: Taux de dropout à appliquer dans la double convolution.
        """
        super(Up, self).__init__()
        
        # Si bilinéaire, utiliser normal convolutions pour réduire le nombre de channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
    
    def forward(self, x1, x2):
        """
        Passe avant du bloc de up-sampling.
        
        Args:
            x1: Tenseur provenant du niveau précédent.
            x2: Tenseur provenant de l'encodeur (skip connection).
            
        Returns:
            Tenseur de sortie après up-sampling et convolutions.
        """
        x1 = self.up(x1)
        
        # Ajuster les dimensions si nécessaire
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        
        # Concaténer x1 et x2
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Couche de convolution de sortie pour U-Net.
    
    Cette couche fait une convolution 1x1 pour obtenir le nombre 
    de canaux souhaité en sortie.
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initialise la couche de convolution de sortie.
        
        Args:
            in_channels: Nombre de canaux d'entrée.
            out_channels: Nombre de canaux de sortie.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Passe avant de la couche de convolution de sortie.
        
        Args:
            x: Tenseur d'entrée.
            
        Returns:
            Tenseur de sortie.
        """
        return self.conv(x)


@ModelRegistry.register("regression_unet")
class RegressionUNet(ForestGapModel):
    """
    Implémentation de U-Net pour les tâches de régression.
    
    Cette architecture est adaptée pour prédire des valeurs continues
    (régression) plutôt que des masques binaires de segmentation.
    """
    
    def __init__(
        self, 
        in_channels=1, 
        out_channels=1, 
        init_features=64,
        bilinear=True,
        dropout_rate=0.2
    ):
        """
        Initialise le modèle U-Net pour la régression.
        
        Args:
            in_channels: Nombre de canaux d'entrée.
            out_channels: Nombre de canaux de sortie.
            init_features: Nombre de features dans la première couche.
            bilinear: Si True, utilise l'interpolation bilinéaire pour l'upsampling.
            dropout_rate: Taux de dropout à appliquer.
        """
        super(RegressionUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.init_features = init_features
        
        # Encodeur
        self.inc = DoubleConv(in_channels, init_features, dropout_rate=dropout_rate)
        self.down1 = Down(init_features, init_features * 2, dropout_rate=dropout_rate)
        self.down2 = Down(init_features * 2, init_features * 4, dropout_rate=dropout_rate)
        self.down3 = Down(init_features * 4, init_features * 8, dropout_rate=dropout_rate)
        factor = 2 if bilinear else 1
        self.down4 = Down(init_features * 8, init_features * 16 // factor, dropout_rate=dropout_rate)
        
        # Décodeur
        self.up1 = Up(init_features * 16, init_features * 8 // factor, bilinear, dropout_rate=dropout_rate)
        self.up2 = Up(init_features * 8, init_features * 4 // factor, bilinear, dropout_rate=dropout_rate)
        self.up3 = Up(init_features * 4, init_features * 2 // factor, bilinear, dropout_rate=dropout_rate)
        self.up4 = Up(init_features * 2, init_features, bilinear, dropout_rate=dropout_rate)
        self.outc = OutConv(init_features, out_channels)
    
    def forward(self, x, threshold=None):
        """
        Passe avant du modèle U-Net pour la régression.
        
        Args:
            x: Tenseur d'entrée [batch_size, in_channels, height, width].
            threshold: Non utilisé dans cette implémentation, présent pour compatibilité.
            
        Returns:
            Tenseur de sortie [batch_size, out_channels, height, width].
        """
        # Encodeur
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Décodeur
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        # Pour la régression, pas de sigmoid à la fin
        # La sortie est directement la valeur prédite
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
        return ["prediction"] 