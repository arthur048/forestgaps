"""
Tests unitaires pour le module de métriques d'évaluation.

Ce module contient des tests pour les métriques utilisées pour évaluer 
les performances des modèles, comme l'IoU, F1, précision, etc.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from forestgaps.training.metrics import (
    SegmentationMetrics,
    iou_metric,
    precision_metric,
    recall_metric,
    f1_metric
)

# ===================================================================================================
# Tests pour les fonctions de métriques individuelles
# ===================================================================================================

def test_iou_metric():
    """Tester la métrique IoU (Intersection over Union)."""
    # Cas simple: prédiction parfaite
    pred = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    assert iou_metric(pred, target).item() == 1.0
    
    # Cas simple: aucun recouvrement
    pred = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    assert iou_metric(pred, target).item() == 0.0
    
    # Cas: recouvrement partiel
    pred = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    # IoU = intersection (2) / union (3) = 2/3
    assert abs(iou_metric(pred, target).item() - 2/3) < 1e-6
    
    # Cas: prédiction à seuil (sigmoid)
    pred = torch.tensor([[0.8, 0.9], [0.2, 0.1]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    # Avec seuil par défaut de 0.5, c'est une prédiction parfaite
    assert iou_metric(pred, target).item() == 1.0

def test_precision_metric():
    """Tester la métrique de précision."""
    # Cas simple: prédiction parfaite
    pred = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    assert precision_metric(pred, target).item() == 1.0
    
    # Cas: faux positifs
    pred = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    # Précision = vrais positifs (2) / (vrais positifs + faux positifs) (2+1) = 2/3
    assert abs(precision_metric(pred, target).item() - 2/3) < 1e-6

def test_recall_metric():
    """Tester la métrique de rappel."""
    # Cas simple: prédiction parfaite
    pred = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    assert recall_metric(pred, target).item() == 1.0
    
    # Cas: faux négatifs
    pred = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    # Rappel = vrais positifs (1) / (vrais positifs + faux négatifs) (1+1) = 1/2
    assert abs(recall_metric(pred, target).item() - 1/2) < 1e-6

def test_f1_metric():
    """Tester la métrique F1."""
    # Cas simple: prédiction parfaite
    pred = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    assert f1_metric(pred, target).item() == 1.0
    
    # Cas: prédiction partiellement correcte
    pred = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    # Précision = 1/2, Rappel = 1/2, F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
    assert abs(f1_metric(pred, target).item() - 0.5) < 1e-6

# ===================================================================================================
# Tests pour la classe SegmentationMetrics
# ===================================================================================================

class TestSegmentationMetrics:
    """Tests pour la classe SegmentationMetrics."""
    
    def setup_method(self):
        """Préparer le contexte pour les tests."""
        self.metrics = SegmentationMetrics(device="cpu")
    
    def test_initialization(self):
        """Tester l'initialisation des métriques."""
        assert self.metrics.device == "cpu"
        assert hasattr(self.metrics, "metrics_by_threshold")
        assert len(self.metrics.metrics_by_threshold) == 0
    
    def test_reset(self):
        """Tester la réinitialisation des métriques."""
        # Mettre à jour les métriques avec des données factices
        pred = torch.tensor([[0.8, 0.2], [0.3, 0.9]])
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.metrics.update(pred, target)
        
        # Vérifier que les métriques ont été mises à jour
        assert len(self.metrics.metrics_by_threshold) > 0
        
        # Réinitialiser les métriques
        self.metrics.reset()
        
        # Vérifier que les métriques ont été réinitialisées
        assert len(self.metrics.metrics_by_threshold) == 0
    
    def test_update(self):
        """Tester la mise à jour des métriques."""
        # Créer des données pour le test
        pred = torch.tensor([[0.8, 0.2], [0.3, 0.9]])
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        
        # Mettre à jour les métriques avec un seuil spécifique
        self.metrics.update(pred, target, threshold=0.5)
        
        # Vérifier que les métriques ont été mises à jour
        assert 0.5 in self.metrics.metrics_by_threshold
        assert "total_iou" in self.metrics.metrics_by_threshold[0.5]
        assert "total_count" in self.metrics.metrics_by_threshold[0.5]
    
    def test_update_by_threshold(self):
        """Tester la mise à jour des métriques par seuil."""
        # Créer des données pour le test
        pred = torch.tensor([[0.8, 0.2], [0.3, 0.9]])
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        
        # Mettre à jour les métriques avec une valeur de seuil
        threshold_value = 10.0  # Valeur de hauteur simulée
        self.metrics.update_by_threshold(pred, target, threshold_value)
        
        # Vérifier que les métriques ont été mises à jour
        assert threshold_value in self.metrics.metrics_by_threshold
    
    def test_compute(self):
        """Tester le calcul des métriques finales."""
        # Créer et mettre à jour des métriques avec différents seuils
        thresholds = [2.0, 5.0, 10.0]
        pred = torch.tensor([[0.8, 0.2], [0.3, 0.9]])
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        
        for threshold in thresholds:
            self.metrics.update_by_threshold(pred, target, threshold)
        
        # Calculer les métriques finales
        results = self.metrics.compute()
        
        # Vérifier la structure des résultats
        assert "mean_iou" in results
        assert "mean_f1" in results
        assert "mean_precision" in results
        assert "mean_recall" in results
        assert "by_threshold" in results
        
        # Vérifier que chaque seuil est présent dans les résultats
        for threshold in thresholds:
            assert threshold in results["by_threshold"]
            assert "iou" in results["by_threshold"][threshold]
            assert "f1" in results["by_threshold"][threshold]
    
    def test_compute_confusion_matrix(self):
        """Tester le calcul de la matrice de confusion."""
        # Créer et mettre à jour des métriques
        pred = torch.tensor([[0.8, 0.2], [0.3, 0.9]])
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        
        self.metrics.update(pred, target, threshold=0.5)
        
        # Calculer la matrice de confusion
        confusion = self.metrics.compute_confusion_matrix()
        
        # Vérifier la structure du résultat
        assert "total" in confusion
        assert "by_threshold" in confusion
        assert "tp" in confusion["total"]
        assert "fp" in confusion["total"]
        assert "fn" in confusion["total"]
        assert "tn" in confusion["total"] 