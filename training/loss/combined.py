"""
Module de fonctions de perte combinées pour l'entraînement des modèles.

Ce module fournit des classes et des fonctions pour les pertes combinées
comme Focal+Dice qui sont particulièrement adaptées à la segmentation 
des trouées forestières.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Any


class FocalLoss(nn.Module):
    """
    Implémentation de la Focal Loss pour gérer le déséquilibre des classes.
    
    Cette perte met davantage l'accent sur les exemples difficiles à classer.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialise la Focal Loss.
        
        Args:
            alpha: Facteur de pondération pour les classes.
            gamma: Facteur de mise à l'échelle qui module l'effet de la correction.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcule la Focal Loss.
        
        Args:
            pred: Tensor des prédictions (probabilités).
            target: Tensor des valeurs cibles.
            
        Returns:
            Valeur de la perte.
        """
        # Clipper les prédictions pour éviter log(0)
        eps = 1e-6
        pred = torch.clamp(pred, eps, 1.0 - eps)
        
        # Calculer la BCE pour chaque pixel
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        # Appliquer la modulation focal
        focal_weight = (target * (1 - pred) ** self.gamma) + ((1 - target) * pred ** self.gamma)
        
        # Pondération optionnelle par alpha
        if self.alpha is not None:
            alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
            focal_weight = focal_weight * alpha_weight
        
        # Calculer la perte finale
        loss = focal_weight * bce
        
        return loss.mean()


class DiceLoss(nn.Module):
    """
    Implémentation de la Dice Loss pour la segmentation.
    
    Cette perte est basée sur le coefficient de Dice/F1 et est particulièrement 
    efficace pour les problèmes de segmentation déséquilibrés.
    """
    
    def __init__(self, smooth: float = 1e-6):
        """
        Initialise la Dice Loss.
        
        Args:
            smooth: Facteur de lissage pour éviter la division par zéro.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcule la Dice Loss.
        
        Args:
            pred: Tensor des prédictions (probabilités).
            target: Tensor des valeurs cibles.
            
        Returns:
            Valeur de la perte.
        """
        # Calculer l'intersection
        intersection = (pred * target).sum()
        
        # Calculer l'union
        union = pred.sum() + target.sum()
        
        # Calculer le coefficient de Dice
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # La perte est le complément du coefficient
        return 1.0 - dice


class CombinedFocalDiceLoss(nn.Module):
    """
    Combinaison de Focal Loss et Dice Loss pour la segmentation des trouées forestières.
    
    Cette combinaison permet de bénéficier des avantages des deux pertes :
    - Focal Loss se concentre sur les exemples difficiles
    - Dice Loss optimise directement les métriques de segmentation
    """
    
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, smooth: float = 1e-6, 
                 threshold_weights: Optional[Dict[float, float]] = None):
        """
        Initialise la perte combinée.
        
        Args:
            alpha: Coefficient de pondération entre les deux pertes.
            gamma: Facteur de mise à l'échelle pour la Focal Loss.
            smooth: Facteur de lissage pour la Dice Loss.
            threshold_weights: Dictionnaire des poids par seuil de hauteur.
        """
        super(CombinedFocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.alpha = alpha
        self.threshold_weights = threshold_weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, threshold: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calcule la perte combinée.
        
        Args:
            pred: Tensor des prédictions (probabilités).
            target: Tensor des valeurs cibles.
            threshold: Tensor des seuils de hauteur (pour pondération).
            
        Returns:
            Valeur de la perte.
        """
        # Vérifier les entrées
        if pred.shape != target.shape:
            raise ValueError(f"Les formes de pred {pred.shape} et target {target.shape} doivent être identiques")
        
        # Calculer les pertes individuelles
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        # Appliquer les poids par seuil si fournis
        if self.threshold_weights is not None and threshold is not None:
            # Créer un tensor de poids
            batch_size = threshold.size(0)
            weights = torch.ones(batch_size, device=pred.device)
            
            # Appliquer les poids en fonction des seuils
            for i in range(batch_size):
                thresh_val = threshold[i].item()
                # Trouver le poids le plus proche
                closest_thresh = min(self.threshold_weights.keys(), 
                                    key=lambda x: abs(x - thresh_val))
                weights[i] = self.threshold_weights[closest_thresh]
            
            # Appliquer les poids
            focal = focal * weights.mean()
            dice = dice * weights.mean()
        
        # Combiner les pertes
        loss = self.alpha * focal + (1 - self.alpha) * dice
        
        return loss


def create_threshold_weights(config, threshold_stats=None):
    """
    Crée des poids pour chaque seuil de hauteur basés sur la distribution des données.
    
    Cette fonction permet de pondérer l'importance des différents seuils de hauteur
    lors de l'entraînement, notamment pour gérer les déséquilibres de classe.
    
    Args:
        config: Configuration contenant les paramètres nécessaires.
        threshold_stats: Statistiques sur la distribution des seuils (optionnel).
        
    Returns:
        Dictionnaire des poids par seuil.
    """
    thresholds = config.thresholds if hasattr(config, 'thresholds') else [5, 10, 15, 20]
    
    # Si aucune statistique n'est fournie, utiliser une distribution uniforme
    if threshold_stats is None:
        # Distribution uniforme
        weights = {threshold: 1.0 for threshold in thresholds}
    else:
        # Calculer les poids inversement proportionnels à la fréquence
        total_samples = sum(threshold_stats.values())
        weights = {}
        
        for threshold in thresholds:
            # Obtenir le nombre d'échantillons pour ce seuil
            samples = threshold_stats.get(threshold, 1)
            
            # Calculer le poids inverse (plus rare = poids plus élevé)
            inverse_freq = total_samples / max(1, samples)
            
            # Normaliser pour avoir des poids entre 0.5 et 2.0
            min_weight = 0.5
            max_weight = 2.0
            
            # Normalisation min-max
            if len(threshold_stats) > 1:
                min_freq = min(threshold_stats.values())
                max_freq = max(threshold_stats.values())
                normalized_weight = min_weight + (max_weight - min_weight) * (inverse_freq - min_freq) / (max_freq - min_freq)
            else:
                normalized_weight = 1.0
            
            weights[threshold] = normalized_weight
    
    # Appliquer un ajustement supplémentaire si spécifié dans la config
    if hasattr(config, 'threshold_weight_adjustment') and config.threshold_weight_adjustment:
        for threshold, adjustment in config.threshold_weight_adjustment.items():
            if threshold in weights:
                weights[threshold] *= adjustment
    
    # Normaliser les poids pour que leur moyenne soit 1.0
    avg_weight = sum(weights.values()) / len(weights)
    weights = {t: w / avg_weight for t, w in weights.items()}
    
    return weights


class AdaptiveLoss(nn.Module):
    """
    Fonction de perte qui s'adapte automatiquement à la distribution des données.
    
    Cette perte ajuste dynamiquement les poids entre différentes sous-pertes
    en fonction des caractéristiques des données d'entraînement.
    """
    
    def __init__(self, loss_functions: List[nn.Module], initial_weights: Optional[List[float]] = None):
        """
        Initialise la perte adaptative.
        
        Args:
            loss_functions: Liste des fonctions de perte à combiner.
            initial_weights: Poids initiaux pour chaque fonction de perte.
        """
        super(AdaptiveLoss, self).__init__()
        self.loss_functions = loss_functions
        
        # Initialiser les poids
        n_losses = len(loss_functions)
        if initial_weights is None:
            initial_weights = [1.0 / n_losses] * n_losses
        
        # Enregistrer les poids comme paramètres pour qu'ils soient appris
        self.weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float))
        
        # Historique des pertes pour le suivi
        self.loss_history = {i: [] for i in range(n_losses)}
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calcule la perte adaptative.
        
        Args:
            pred: Tensor des prédictions.
            target: Tensor des valeurs cibles.
            **kwargs: Arguments supplémentaires pour les fonctions de perte.
            
        Returns:
            Valeur de la perte.
        """
        # Calculer chaque perte individuellement
        losses = []
        for i, loss_fn in enumerate(self.loss_functions):
            loss_val = loss_fn(pred, target, **kwargs)
            losses.append(loss_val)
            self.loss_history[i].append(loss_val.item())
        
        # Appliquer softmax aux poids pour qu'ils somment à 1
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # Combiner les pertes selon les poids
        total_loss = sum(w * l for w, l in zip(normalized_weights, losses))
        
        return total_loss
    
    def get_weights(self) -> torch.Tensor:
        """
        Renvoie les poids actuels des fonctions de perte.
        
        Returns:
            Tensor des poids normalisés.
        """
        return F.softmax(self.weights, dim=0)
    
    def get_loss_history(self) -> Dict[int, List[float]]:
        """
        Renvoie l'historique des pertes.
        
        Returns:
            Dictionnaire contenant l'historique des pertes pour chaque fonction.
        """
        return self.loss_history 