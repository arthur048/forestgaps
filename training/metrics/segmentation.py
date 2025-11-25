"""
Module de métriques de segmentation pour l'évaluation des modèles.

Ce module fournit des classes et des fonctions pour calculer les métriques
d'évaluation spécifiques à la segmentation des trouées forestières.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union


def iou_metric(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calcule l'Intersection over Union (IoU) entre les prédictions et les cibles.
    
    Args:
        pred: Tensor des prédictions.
        target: Tensor des valeurs cibles.
        smooth: Facteur de lissage pour éviter la division par zéro.
        
    Returns:
        Score IoU.
    """
    # Assurez-vous que les tenseurs sont du même type
    pred = pred.float()
    target = target.float()
    
    # Binariser les prédictions si nécessaire
    if pred.shape != target.shape:
        pred = (pred > 0.5).float()
    
    # Calculer l'intersection et l'union
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    # Calculer l'IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


class SegmentationMetrics:
    """
    Classe pour calculer et stocker les métriques de segmentation.
    
    Cette classe permet de suivre différentes métriques de segmentation 
    (précision, rappel, F1-score, IoU) globalement et par seuil de hauteur.
    """
    
    def __init__(self, device=None):
        """
        Initialise les compteurs pour les métriques de segmentation.
        
        Args:
            device: Dispositif sur lequel effectuer les calculs (CPU/GPU).
        """
        self.device = device
        self.reset()
    
    def reset(self):
        """
        Réinitialise toutes les métriques.
        """
        # Métriques globales
        self.tp = 0.0  # True positives
        self.fp = 0.0  # False positives
        self.tn = 0.0  # True negatives
        self.fn = 0.0  # False negatives
        self.n_updates = 0
        
        # Métriques par seuil de hauteur
        self.threshold_metrics = {}
        
        # Stockage pour les métriques de confusion
        self.confusion_data = {
            'tp': 0.0,
            'fp': 0.0,
            'tn': 0.0,
            'fn': 0.0,
            'total_pixels': 0.0,
            'total_gap_pixels': 0.0,
            'total_non_gap_pixels': 0.0
        }
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
        """
        Met à jour les métriques avec de nouvelles prédictions.
        
        Args:
            pred: Tensor des prédictions.
            target: Tensor des valeurs cibles.
            threshold: Seuil pour binariser les prédictions.
            
        Returns:
            Dictionnaire des métriques calculées.
        """
        # Assurez-vous que les tenseurs sont du même type
        pred = pred.float()
        target = target.float()
        
        # Binariser les prédictions si nécessaire
        if pred.dim() > 1 and pred.shape[1] > 1:
            # Si multi-classes, prendre l'argmax
            pred = torch.argmax(pred, dim=1).float()
        elif pred.dim() > 1:
            # Si une seule classe, appliquer le seuil
            pred = (pred > threshold).float()
        else:
            # Si déjà binarisé, s'assurer que c'est un float
            pred = (pred > threshold).float()
        
        # Calculer les métriques
        tp = (pred * target).sum().item()
        fp = (pred * (1 - target)).sum().item()
        tn = ((1 - pred) * (1 - target)).sum().item()
        fn = ((1 - pred) * target).sum().item()
        
        # Mettre à jour les compteurs
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn
        self.n_updates += 1
        
        # Calculer les métriques actuelles
        return self._calculate_metrics(tp, fp, tn, fn)
    
    def update_by_threshold(self, pred: torch.Tensor, target: torch.Tensor, threshold_value: float) -> Dict[str, Dict[str, float]]:
        """
        Met à jour les métriques spécifiques à un seuil de hauteur.
        
        Args:
            pred: Tensor des prédictions.
            target: Tensor des valeurs cibles.
            threshold_value: Valeur du seuil de hauteur.
            
        Returns:
            Dictionnaire des métriques par seuil.
        """
        # Assurez-vous que les tenseurs sont du même type
        pred = pred.float()
        target = target.float()
        
        # Binariser les prédictions
        pred_bin = (pred > 0.5).float()
        
        # Calculer les métriques pour ce seuil
        tp = (pred_bin * target).sum().item()
        fp = (pred_bin * (1 - target)).sum().item()
        tn = ((1 - pred_bin) * (1 - target)).sum().item()
        fn = ((1 - pred_bin) * target).sum().item()
        
        # Mettre à jour les métriques pour ce seuil
        if threshold_value not in self.threshold_metrics:
            self.threshold_metrics[threshold_value] = {
                'tp': 0.0, 'fp': 0.0, 'tn': 0.0, 'fn': 0.0, 'n_updates': 0
            }
        
        self.threshold_metrics[threshold_value]['tp'] += tp
        self.threshold_metrics[threshold_value]['fp'] += fp
        self.threshold_metrics[threshold_value]['tn'] += tn
        self.threshold_metrics[threshold_value]['fn'] += fn
        self.threshold_metrics[threshold_value]['n_updates'] += 1
        
        # Calculer les métriques pour tous les seuils
        threshold_results = {}
        for thresh, values in self.threshold_metrics.items():
            metrics = self._calculate_metrics(values['tp'], values['fp'], values['tn'], values['fn'])
            threshold_results[thresh] = metrics
        
        return threshold_results
    
    def compute(self) -> Dict[str, float]:
        """
        Calcule les métriques finales à partir des compteurs accumulés.
        
        Returns:
            Dictionnaire des métriques calculées.
        """
        # Calculer les métriques globales
        metrics = self._calculate_metrics(self.tp, self.fp, self.tn, self.fn)
        
        # Calculer les métriques par seuil
        threshold_metrics = {}
        for threshold, values in self.threshold_metrics.items():
            threshold_metrics[threshold] = self._calculate_metrics(
                values['tp'], values['fp'], values['tn'], values['fn']
            )
        
        # Ajouter les métriques par seuil aux métriques globales
        metrics['threshold_metrics'] = threshold_metrics
        
        return metrics
    
    def compute_confusion_matrix(self) -> Dict[str, float]:
        """
        Calcule les métriques de confusion pour l'analyse des erreurs.
        
        Returns:
            Dictionnaire des métriques de confusion.
        """
        total_pixels = self.tp + self.fp + self.tn + self.fn
        total_gap_pixels = self.tp + self.fn
        total_non_gap_pixels = self.fp + self.tn
        
        # Calculer les métriques de confusion
        if total_pixels > 0:
            self.confusion_data = {
                'tp': self.tp,
                'fp': self.fp,
                'tn': self.tn,
                'fn': self.fn,
                'total_pixels': total_pixels,
                'total_gap_pixels': total_gap_pixels,
                'total_non_gap_pixels': total_non_gap_pixels,
                'tp_rate': self.tp / total_gap_pixels if total_gap_pixels > 0 else 0,
                'fp_rate': self.fp / total_non_gap_pixels if total_non_gap_pixels > 0 else 0,
                'tn_rate': self.tn / total_non_gap_pixels if total_non_gap_pixels > 0 else 0,
                'fn_rate': self.fn / total_gap_pixels if total_gap_pixels > 0 else 0,
                'gap_proportion': total_gap_pixels / total_pixels,
                'non_gap_proportion': total_non_gap_pixels / total_pixels
            }
        
        return self.confusion_data
    
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
        
        # Calculer les métriques
        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)
        f1 = 2 * precision * recall / (precision + recall + smooth)
        
        # IoU (Jaccard index)
        iou = tp / (tp + fp + fn + smooth)
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)
        
        # Mesure F2 (met l'accent sur le rappel)
        f2 = (1 + 2**2) * precision * recall / ((2**2) * precision + recall + smooth)
        
        # Mesure F0.5 (met l'accent sur la précision)
        f05 = (1 + 0.5**2) * precision * recall / ((0.5**2) * precision + recall + smooth)
        
        # Dice coefficient (équivalent à F1-score)
        dice = 2 * tp / (2 * tp + fp + fn + smooth)
        
        # Taux d'erreur
        error_rate = (fp + fn) / (tp + tn + fp + fn + smooth)
        
        # MCC (Matthews Correlation Coefficient)
        mcc_num = tp * tn - fp * fn
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + smooth)
        mcc = mcc_num / mcc_den
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'accuracy': accuracy,
            'f2': f2,
            'f05': f05,
            'dice': dice,
            'error_rate': error_rate,
            'mcc': mcc
        } 