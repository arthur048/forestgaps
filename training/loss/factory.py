"""
Module factory pour la création des fonctions de perte.

Ce module fournit des fonctions pour créer différentes fonctions de perte
à partir de la configuration.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union

from .combined import CombinedFocalDiceLoss, FocalLoss, DiceLoss, AdaptiveLoss


def create_loss_function(config: Any) -> nn.Module:
    """
    Crée une fonction de perte à partir de la configuration.
    
    Args:
        config: Configuration contenant les paramètres de la fonction de perte.
        
    Returns:
        Fonction de perte correspondant à la configuration.
    """
    loss_type = getattr(config, 'loss_type', 'combined_focal_dice')
    
    if loss_type == 'combined_focal_dice':
        # Paramètres optionnels
        alpha = getattr(config, 'focal_alpha', 0.5)
        gamma = getattr(config, 'focal_gamma', 2.0)
        smooth = getattr(config, 'dice_smooth', 1e-6)
        
        # Créer la fonction de perte
        return CombinedFocalDiceLoss(alpha=alpha, gamma=gamma, smooth=smooth)
    
    elif loss_type == 'focal':
        alpha = getattr(config, 'focal_alpha', 0.25)
        gamma = getattr(config, 'focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'dice':
        smooth = getattr(config, 'dice_smooth', 1e-6)
        return DiceLoss(smooth=smooth)
    
    elif loss_type == 'bce':
        return nn.BCELoss()
    
    elif loss_type == 'adaptive':
        # Créer une combinaison adaptative des pertes
        loss_functions = []
        initial_weights = getattr(config, 'adaptive_weights', None)
        
        # Ajouter les pertes individuelles
        if getattr(config, 'use_focal', True):
            alpha = getattr(config, 'focal_alpha', 0.25)
            gamma = getattr(config, 'focal_gamma', 2.0)
            loss_functions.append(FocalLoss(alpha=alpha, gamma=gamma))
        
        if getattr(config, 'use_dice', True):
            smooth = getattr(config, 'dice_smooth', 1e-6)
            loss_functions.append(DiceLoss(smooth=smooth))
        
        if getattr(config, 'use_bce', False):
            loss_functions.append(nn.BCELoss())
        
        # S'assurer qu'il y a au moins une fonction de perte
        if not loss_functions:
            loss_functions = [FocalLoss(), DiceLoss()]
        
        return AdaptiveLoss(loss_functions, initial_weights)
    
    else:
        raise ValueError(f"Type de fonction de perte non reconnu: {loss_type}")


def create_loss_with_threshold_weights(config: Any, threshold_stats: Optional[Dict[float, int]] = None) -> nn.Module:
    """
    Crée une fonction de perte avec pondération par seuil de hauteur.
    
    Args:
        config: Configuration contenant les paramètres nécessaires.
        threshold_stats: Statistiques sur la distribution des seuils.
        
    Returns:
        Fonction de perte avec pondération par seuil.
    """
    from .combined import create_threshold_weights
    
    # Créer les poids par seuil
    threshold_weights = create_threshold_weights(config, threshold_stats)
    
    # Créer la fonction de perte avec les poids
    loss_fn = create_loss_function(config)
    
    # Si c'est une CombinedFocalDiceLoss, appliquer les poids directement
    if isinstance(loss_fn, CombinedFocalDiceLoss):
        loss_fn.threshold_weights = threshold_weights
    
    # Pour les autres types, encapsuler dans une fonction wrapper
    else:
        original_forward = loss_fn.forward
        
        def forward_with_weights(pred, target, threshold=None):
            loss = original_forward(pred, target)
            
            if threshold is not None:
                batch_size = threshold.size(0)
                weights = torch.ones(batch_size, device=pred.device)
                
                for i in range(batch_size):
                    thresh_val = threshold[i].item()
                    closest_thresh = min(threshold_weights.keys(), 
                                        key=lambda x: abs(x - thresh_val))
                    weights[i] = threshold_weights[closest_thresh]
                
                loss = loss * weights.mean()
            
            return loss
        
        loss_fn.forward = forward_with_weights
    
    return loss_fn 