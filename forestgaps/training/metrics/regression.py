"""
Module de métriques pour l'évaluation des modèles de régression.

Ce module fournit des fonctions et des classes pour calculer des métriques
spécifiques à l'évaluation des modèles de régression, comme MSE, MAE, RMSE, etc.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union


def mse_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calcule l'erreur quadratique moyenne (MSE) entre les prédictions et les cibles.
    
    Args:
        pred: Tenseur des prédictions [batch_size, channels, height, width].
        target: Tenseur des cibles [batch_size, channels, height, width].
        
    Returns:
        Erreur quadratique moyenne.
    """
    return torch.mean((pred - target) ** 2)


def mae_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calcule l'erreur absolue moyenne (MAE) entre les prédictions et les cibles.
    
    Args:
        pred: Tenseur des prédictions [batch_size, channels, height, width].
        target: Tenseur des cibles [batch_size, channels, height, width].
        
    Returns:
        Erreur absolue moyenne.
    """
    return torch.mean(torch.abs(pred - target))


def rmse_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calcule la racine de l'erreur quadratique moyenne (RMSE) entre les prédictions et les cibles.
    
    Args:
        pred: Tenseur des prédictions [batch_size, channels, height, width].
        target: Tenseur des cibles [batch_size, channels, height, width].
        
    Returns:
        Racine de l'erreur quadratique moyenne.
    """
    return torch.sqrt(mse_metric(pred, target))


def r2_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calcule le coefficient de détermination (R²) entre les prédictions et les cibles.
    
    Args:
        pred: Tenseur des prédictions [batch_size, channels, height, width].
        target: Tenseur des cibles [batch_size, channels, height, width].
        
    Returns:
        Coefficient de détermination.
    """
    # Mettre à plat les tenseurs
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    # Calculer les statistiques
    target_mean = torch.mean(target_flat)
    ss_tot = torch.sum((target_flat - target_mean) ** 2)
    ss_res = torch.sum((target_flat - pred_flat) ** 2)
    
    # Éviter la division par zéro
    if ss_tot == 0:
        return torch.tensor(0.0, device=pred.device)
    
    return 1 - (ss_res / ss_tot)


class RegressionMetrics:
    """
    Classe pour calculer, accumuler et rapporter les métriques de régression.
    
    Cette classe permet de suivre les métriques comme MSE, MAE, RMSE et R²
    au cours de l'entraînement ou de l'évaluation d'un modèle de régression.
    """
    
    def __init__(self, device=None):
        """
        Initialise le système de suivi des métriques de régression.
        
        Args:
            device: Dispositif sur lequel effectuer les calculs (None pour auto-détection).
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()
    
    def reset(self):
        """
        Réinitialise toutes les métriques accumulées.
        """
        self.mse_sum = 0.0
        self.mae_sum = 0.0
        self.r2_sum = 0.0
        self.batch_count = 0
        self.pixel_count = 0
        
        # Métriques par seuil pour les modèles conditionnés
        self.mse_by_threshold = {}
        self.mae_by_threshold = {}
        self.r2_by_threshold = {}
        self.batch_count_by_threshold = {}
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, threshold: Optional[torch.Tensor] = None):
        """
        Met à jour les métriques avec un nouveau batch de prédictions et cibles.
        
        Args:
            pred: Tenseur des prédictions [batch_size, channels, height, width].
            target: Tenseur des cibles [batch_size, channels, height, width].
            threshold: Tenseur des seuils [batch_size, 1] pour les métriques par seuil.
        """
        # Calcul des métriques
        mse = mse_metric(pred, target).item()
        mae = mae_metric(pred, target).item()
        r2 = r2_metric(pred, target).item()
        
        # Mise à jour des métriques globales
        self.mse_sum += mse * pred.size(0)  # Pondéré par la taille du batch
        self.mae_sum += mae * pred.size(0)
        self.r2_sum += r2 * pred.size(0)
        self.batch_count += 1
        self.pixel_count += pred.size(0) * pred.size(2) * pred.size(3)
        
        # Mise à jour des métriques par seuil si un seuil est fourni
        if threshold is not None:
            thresholds = threshold.cpu().numpy().flatten()
            for i, thresh in enumerate(thresholds):
                thresh_key = float(thresh)
                
                # Indices pour sélectionner un seul élément du batch
                batch_indices = [i]
                
                # Extraire les prédictions et cibles pour cet élément
                pred_i = pred[i].unsqueeze(0)
                target_i = target[i].unsqueeze(0)
                
                # Calculer les métriques pour cet élément
                mse_i = mse_metric(pred_i, target_i).item()
                mae_i = mae_metric(pred_i, target_i).item()
                r2_i = r2_metric(pred_i, target_i).item()
                
                # Mettre à jour les dictionnaires des métriques par seuil
                if thresh_key not in self.mse_by_threshold:
                    self.mse_by_threshold[thresh_key] = 0.0
                    self.mae_by_threshold[thresh_key] = 0.0
                    self.r2_by_threshold[thresh_key] = 0.0
                    self.batch_count_by_threshold[thresh_key] = 0
                
                self.mse_by_threshold[thresh_key] += mse_i
                self.mae_by_threshold[thresh_key] += mae_i
                self.r2_by_threshold[thresh_key] += r2_i
                self.batch_count_by_threshold[thresh_key] += 1
    
    def compute(self) -> Dict[str, float]:
        """
        Calcule et retourne toutes les métriques accumulées.
        
        Returns:
            Dictionnaire contenant les métriques calculées.
        """
        if self.batch_count == 0:
            return {
                'mse': 0.0,
                'rmse': 0.0,
                'mae': 0.0,
                'r2': 0.0
            }
        
        # Calculer les métriques moyennes
        avg_mse = self.mse_sum / self.batch_count
        avg_mae = self.mae_sum / self.batch_count
        avg_r2 = self.r2_sum / self.batch_count
        
        # Calculer la RMSE
        rmse = np.sqrt(avg_mse)
        
        # Créer le dictionnaire des résultats
        results = {
            'mse': avg_mse,
            'rmse': rmse,
            'mae': avg_mae,
            'r2': avg_r2
        }
        
        # Ajouter les métriques par seuil
        for thresh, count in self.batch_count_by_threshold.items():
            if count > 0:
                thresh_mse = self.mse_by_threshold[thresh] / count
                thresh_mae = self.mae_by_threshold[thresh] / count
                thresh_r2 = self.r2_by_threshold[thresh] / count
                thresh_rmse = np.sqrt(thresh_mse)
                
                results[f'mse_threshold_{thresh}'] = thresh_mse
                results[f'rmse_threshold_{thresh}'] = thresh_rmse
                results[f'mae_threshold_{thresh}'] = thresh_mae
                results[f'r2_threshold_{thresh}'] = thresh_r2
        
        return results 