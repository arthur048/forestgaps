"""
Module de fonctions de perte pour les modèles de régression.

Ce module fournit des implémentations de fonctions de perte adaptées
aux tâches de régression sur des données forestières.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, List


class MSELoss(nn.Module):
    """
    Implémentation de la fonction de perte Mean Squared Error (MSE).
    
    Cette fonction de perte calcule l'erreur quadratique moyenne entre
    les prédictions et les cibles.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialise la fonction de perte MSE.
        
        Args:
            reduction: Mode de réduction ('mean', 'sum', 'none').
        """
        super(MSELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, threshold: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calcule la perte MSE entre les prédictions et les cibles.
        
        Args:
            pred: Tenseur des prédictions [batch_size, channels, height, width].
            target: Tenseur des cibles [batch_size, channels, height, width].
            threshold: Non utilisé, présent pour compatibilité.
            
        Returns:
            Perte MSE.
        """
        return F.mse_loss(pred, target, reduction=self.reduction)


class MAELoss(nn.Module):
    """
    Implémentation de la fonction de perte Mean Absolute Error (MAE).
    
    Cette fonction de perte calcule l'erreur absolue moyenne entre
    les prédictions et les cibles.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialise la fonction de perte MAE.
        
        Args:
            reduction: Mode de réduction ('mean', 'sum', 'none').
        """
        super(MAELoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, threshold: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calcule la perte MAE entre les prédictions et les cibles.
        
        Args:
            pred: Tenseur des prédictions [batch_size, channels, height, width].
            target: Tenseur des cibles [batch_size, channels, height, width].
            threshold: Non utilisé, présent pour compatibilité.
            
        Returns:
            Perte MAE.
        """
        return F.l1_loss(pred, target, reduction=self.reduction)


class HuberLoss(nn.Module):
    """
    Implémentation de la fonction de perte Huber (smooth L1).
    
    Cette fonction de perte combine les avantages de MSE et MAE,
    en étant moins sensible aux outliers que MSE.
    """
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        """
        Initialise la fonction de perte Huber.
        
        Args:
            beta: Paramètre de transition entre L1 et L2.
            reduction: Mode de réduction ('mean', 'sum', 'none').
        """
        super(HuberLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, threshold: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calcule la perte Huber entre les prédictions et les cibles.
        
        Args:
            pred: Tenseur des prédictions [batch_size, channels, height, width].
            target: Tenseur des cibles [batch_size, channels, height, width].
            threshold: Non utilisé, présent pour compatibilité.
            
        Returns:
            Perte Huber.
        """
        return F.smooth_l1_loss(pred, target, beta=self.beta, reduction=self.reduction)


class CombinedRegressionLoss(nn.Module):
    """
    Implémentation d'une fonction de perte combinée pour la régression.
    
    Cette fonction combine MSE et MAE avec des poids configurables,
    offrant un bon compromis entre les deux types de pertes.
    """
    
    def __init__(self, mse_weight: float = 0.5, mae_weight: float = 0.5, reduction: str = 'mean'):
        """
        Initialise la fonction de perte combinée.
        
        Args:
            mse_weight: Poids de la composante MSE.
            mae_weight: Poids de la composante MAE.
            reduction: Mode de réduction ('mean', 'sum', 'none').
        """
        super(CombinedRegressionLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.reduction = reduction
        
        # Sous-fonctions de perte
        self.mse_loss = MSELoss(reduction=reduction)
        self.mae_loss = MAELoss(reduction=reduction)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, threshold: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calcule la perte combinée entre les prédictions et les cibles.
        
        Args:
            pred: Tenseur des prédictions [batch_size, channels, height, width].
            target: Tenseur des cibles [batch_size, channels, height, width].
            threshold: Non utilisé, présent pour compatibilité.
            
        Returns:
            Perte combinée.
        """
        mse = self.mse_loss(pred, target)
        mae = self.mae_loss(pred, target)
        
        return self.mse_weight * mse + self.mae_weight * mae


class RegressionLossWithWeights(nn.Module):
    """
    Fonction de perte avec pondération spatiale pour la régression.
    
    Cette fonction permet de donner plus de poids à certaines régions spatiales
    lors du calcul de la perte, ce qui peut être utile pour se concentrer 
    sur les zones plus importantes ou difficiles.
    """
    
    def __init__(self, base_loss: str = 'mse', use_spatial_weights: bool = True, reduction: str = 'mean'):
        """
        Initialise la fonction de perte avec pondération.
        
        Args:
            base_loss: Type de perte de base ('mse', 'mae', 'huber', 'combined').
            use_spatial_weights: Si True, utilise une pondération spatiale.
            reduction: Mode de réduction ('mean', 'sum', 'none').
        """
        super(RegressionLossWithWeights, self).__init__()
        self.use_spatial_weights = use_spatial_weights
        self.reduction = reduction
        
        # Créer la fonction de perte de base
        if base_loss == 'mse':
            self.base_loss = MSELoss(reduction='none')
        elif base_loss == 'mae':
            self.base_loss = MAELoss(reduction='none')
        elif base_loss == 'huber':
            self.base_loss = HuberLoss(reduction='none')
        elif base_loss == 'combined':
            self.base_loss = CombinedRegressionLoss(reduction='none')
        else:
            raise ValueError(f"Type de perte non reconnu: {base_loss}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, threshold: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calcule la perte pondérée entre les prédictions et les cibles.
        
        Args:
            pred: Tenseur des prédictions [batch_size, channels, height, width].
            target: Tenseur des cibles [batch_size, channels, height, width].
            threshold: Utilisé pour calculer les poids si use_spatial_weights=True.
            
        Returns:
            Perte pondérée.
        """
        # Calculer la perte de base (sans réduction)
        base_loss = self.base_loss(pred, target)
        
        # Appliquer la pondération spatiale si demandé
        if self.use_spatial_weights and threshold is not None:
            # Créer une carte de poids basée sur le seuil
            # Plus le seuil est élevé, plus le poids est important
            batch_size = pred.size(0)
            weights = []
            
            for i in range(batch_size):
                thresh = threshold[i].item()
                # Normaliser le seuil entre 0.5 et 2.0 pour la pondération
                weight_factor = 0.5 + 1.5 * (thresh / 15.0)  # Supposant que thresh max = 15
                weights.append(weight_factor)
            
            # Convertir en tenseur et ajouter les dimensions
            weights = torch.tensor(weights, device=pred.device)
            weights = weights.view(batch_size, 1, 1, 1)
            
            # Appliquer les poids
            weighted_loss = base_loss * weights
        else:
            weighted_loss = base_loss
        
        # Appliquer la réduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:  # 'none'
            return weighted_loss 