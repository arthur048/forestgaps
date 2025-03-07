"""
Module de métriques de classification pour l'évaluation des modèles.

Ce module fournit des classes et des fonctions pour calculer les métriques
d'évaluation spécifiques à la classification des trouées forestières par seuil de hauteur.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union


class ThresholdMetrics:
    """
    Classe pour calculer et stocker les métriques par seuil de hauteur.
    
    Cette classe permet de suivre les performances de segmentation
    en fonction des différents seuils de hauteur des arbres.
    """
    
    def __init__(self, thresholds: List[float]):
        """
        Initialise les métriques par seuil.
        
        Args:
            thresholds: Liste des seuils de hauteur à évaluer.
        """
        self.thresholds = thresholds
        self.reset()
    
    def reset(self):
        """
        Réinitialise toutes les métriques.
        """
        self.metrics = {threshold: {
            'tp': 0.0, 'fp': 0.0, 'tn': 0.0, 'fn': 0.0,
            'n_samples': 0, 'n_pixels': 0.0, 'gap_pixels': 0.0
        } for threshold in self.thresholds}
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, threshold: float, 
               threshold_value: float) -> Dict[str, float]:
        """
        Met à jour les métriques pour un seuil de hauteur spécifique.
        
        Args:
            pred: Tensor des prédictions.
            target: Tensor des valeurs cibles.
            threshold: Seuil pour binariser les prédictions.
            threshold_value: Valeur du seuil de hauteur.
            
        Returns:
            Dictionnaire des métriques calculées pour ce seuil.
        """
        # S'assurer que le seuil est dans la liste
        if threshold_value not in self.thresholds:
            raise ValueError(f"Le seuil {threshold_value} n'est pas dans la liste des seuils")
        
        # Binariser les prédictions
        pred_bin = (pred > threshold).float()
        
        # Calculer les métriques
        tp = (pred_bin * target).sum().item()
        fp = (pred_bin * (1 - target)).sum().item()
        tn = ((1 - pred_bin) * (1 - target)).sum().item()
        fn = ((1 - pred_bin) * target).sum().item()
        
        # Mettre à jour les compteurs pour ce seuil
        self.metrics[threshold_value]['tp'] += tp
        self.metrics[threshold_value]['fp'] += fp
        self.metrics[threshold_value]['tn'] += tn
        self.metrics[threshold_value]['fn'] += fn
        self.metrics[threshold_value]['n_samples'] += 1
        self.metrics[threshold_value]['n_pixels'] += (tp + fp + tn + fn)
        self.metrics[threshold_value]['gap_pixels'] += (tp + fn)
        
        # Calculer les métriques actuelles pour ce seuil
        return self._calculate_metrics(
            self.metrics[threshold_value]['tp'],
            self.metrics[threshold_value]['fp'],
            self.metrics[threshold_value]['tn'],
            self.metrics[threshold_value]['fn']
        )
    
    def compute(self) -> Dict[float, Dict[str, float]]:
        """
        Calcule les métriques finales pour tous les seuils.
        
        Returns:
            Dictionnaire des métriques par seuil.
        """
        results = {}
        for threshold, values in self.metrics.items():
            if values['n_samples'] > 0:
                results[threshold] = self._calculate_metrics(
                    values['tp'], values['fp'], values['tn'], values['fn']
                )
                # Ajouter des statistiques supplémentaires
                results[threshold]['n_samples'] = values['n_samples']
                results[threshold]['gap_ratio'] = values['gap_pixels'] / values['n_pixels'] if values['n_pixels'] > 0 else 0
        
        return results
    
    def _calculate_metrics(self, tp: float, fp: float, tn: float, fn: float) -> Dict[str, float]:
        """
        Calcule diverses métriques à partir des compteurs.
        
        Args:
            tp: Nombre de vrais positifs.
            fp: Nombre de faux positifs.
            tn: Nombre de vrais négatifs.
            fn: Nombre de faux négatifs.
            
        Returns:
            Dictionnaire des métriques calculées.
        """
        # Éviter les divisions par zéro
        smooth = 1e-6
        
        # Calculer les métriques de base
        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)
        specificity = tn / (tn + fp + smooth)
        
        # F-scores
        f1 = 2 * precision * recall / (precision + recall + smooth)
        f2 = (1 + 4) * precision * recall / (4 * precision + recall + smooth)
        f05 = (1 + 0.25) * precision * recall / (0.25 * precision + recall + smooth)
        
        # Autres métriques
        iou = tp / (tp + fp + fn + smooth)
        accuracy = (tp + tn) / (tp + fp + tn + fn + smooth)
        balanced_accuracy = (recall + specificity) / 2
        
        return {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'f2': f2,
            'f05': f05,
            'iou': iou,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convertit les métriques calculées en DataFrame pandas pour faciliter l'analyse.
        
        Returns:
            DataFrame contenant les métriques par seuil.
        """
        metrics_dict = self.compute()
        
        # Créer un DataFrame pour chaque métrique
        data = []
        for threshold, metrics in metrics_dict.items():
            metrics['threshold'] = threshold
            data.append(metrics)
        
        df = pd.DataFrame(data)
        
        # Trier par seuil
        if not df.empty:
            df = df.sort_values('threshold')
        
        return df


class MetricAnalyzer:
    """
    Classe pour analyser les métriques et identifier les tendances.
    
    Cette classe permet d'analyser les résultats des métriques par seuil
    et d'identifier les points forts et faibles du modèle.
    """
    
    def __init__(self, metrics_data: Dict[float, Dict[str, float]]):
        """
        Initialise l'analyseur avec les métriques calculées.
        
        Args:
            metrics_data: Dictionnaire des métriques par seuil.
        """
        self.metrics_data = metrics_data
        self.thresholds = sorted(metrics_data.keys())
    
    def find_best_threshold(self, metric='f1') -> Tuple[float, float]:
        """
        Trouve le seuil avec la meilleure valeur pour une métrique donnée.
        
        Args:
            metric: Nom de la métrique à optimiser.
            
        Returns:
            Tuple contenant le meilleur seuil et la valeur de la métrique.
        """
        best_threshold = self.thresholds[0]
        best_value = self.metrics_data[best_threshold].get(metric, 0)
        
        for threshold in self.thresholds:
            current_value = self.metrics_data[threshold].get(metric, 0)
            if current_value > best_value:
                best_value = current_value
                best_threshold = threshold
        
        return best_threshold, best_value
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Génère un résumé des performances pour tous les seuils.
        
        Returns:
            Dictionnaire contenant le résumé des performances.
        """
        # Trouver les meilleurs seuils pour différentes métriques
        best_thresholds = {}
        for metric in ['f1', 'precision', 'recall', 'iou', 'balanced_accuracy']:
            best_thresholds[metric] = self.find_best_threshold(metric)
        
        # Calculer des statistiques globales
        avg_metrics = {
            metric: np.mean([data.get(metric, 0) for data in self.metrics_data.values()])
            for metric in ['f1', 'precision', 'recall', 'iou', 'accuracy']
        }
        
        # Identifier les tendances
        trends = {}
        for metric in ['f1', 'precision', 'recall']:
            values = [data.get(metric, 0) for threshold, data in sorted(self.metrics_data.items())]
            if len(values) > 1:
                trend = 'increasing' if values[-1] > values[0] else 'decreasing'
                trends[metric] = trend
        
        return {
            'best_thresholds': best_thresholds,
            'avg_metrics': avg_metrics,
            'trends': trends
        } 