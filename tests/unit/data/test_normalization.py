"""
Tests unitaires pour le module de normalisation des données.

Ce module contient des tests pour les fonctionnalités de normalisation de données,
comme le calcul des statistiques de normalisation et l'application de la normalisation.
"""

import os
import tempfile
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

from forestgaps_dl.data.normalization import (
    compute_normalization_stats,
    NormalizationLayer,
    apply_normalization
)

# ===================================================================================================
# Tests pour le calcul des statistiques de normalisation
# ===================================================================================================

class TestComputeNormalizationStats:
    """Tests pour le calcul des statistiques de normalisation."""
    
    def setup_method(self):
        """Préparer l'environnement de test."""
        # Créer des données synthétiques
        self.mock_data = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32)
        ]
        
    @patch('os.listdir')
    @patch('numpy.load')
    def test_compute_stats(self, mock_load, mock_listdir):
        """Tester le calcul des statistiques de normalisation."""
        # Configurer les mocks
        mock_listdir.return_value = ['tile1_dsm.npy', 'tile2_dsm.npy', 'tile3_dsm.npy']
        mock_load.side_effect = self.mock_data
        
        # Appeler la fonction avec un répertoire temporaire
        with tempfile.TemporaryDirectory() as temp_dir:
            stats = compute_normalization_stats(temp_dir, save_path=None)
            
            # Vérifier les statistiques calculées
            assert abs(stats['min'] - 1.0) < 1e-6
            assert abs(stats['max'] - 12.0) < 1e-6
            assert abs(stats['mean'] - 6.5) < 1e-6
            assert 'std' in stats
    
    @patch('os.listdir')
    @patch('numpy.load')
    @patch('torch.save')
    def test_save_stats(self, mock_save, mock_load, mock_listdir):
        """Tester la sauvegarde des statistiques de normalisation."""
        # Configurer les mocks
        mock_listdir.return_value = ['tile1_dsm.npy', 'tile2_dsm.npy', 'tile3_dsm.npy']
        mock_load.side_effect = self.mock_data
        
        # Appeler la fonction avec un répertoire temporaire et un chemin de sauvegarde
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, 'stats.pt')
            stats = compute_normalization_stats(temp_dir, save_path=save_path)
            
            # Vérifier que torch.save a été appelé
            mock_save.assert_called_once()
            
# ===================================================================================================
# Tests pour la couche de normalisation
# ===================================================================================================

class TestNormalizationLayer:
    """Tests pour la couche de normalisation."""
    
    def test_min_max_normalization(self):
        """Tester la normalisation min-max."""
        # Créer des données de test
        input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Créer la couche de normalisation
        norm_layer = NormalizationLayer(
            method='min-max',
            stats={'min': 1.0, 'max': 4.0}
        )
        
        # Appliquer la normalisation
        output = norm_layer(input_tensor)
        
        # Vérifier la sortie
        expected = torch.tensor([[0.0, 1/3], [2/3, 1.0]])
        assert torch.allclose(output, expected, atol=1e-6)
    
    def test_z_score_normalization(self):
        """Tester la normalisation z-score."""
        # Créer des données de test
        input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Statistiques (mean=2.5, std=1.118)
        mean = 2.5
        std = 1.118  # Approximation de l'écart-type réel
        
        # Créer la couche de normalisation
        norm_layer = NormalizationLayer(
            method='z-score',
            stats={'mean': mean, 'std': std}
        )
        
        # Appliquer la normalisation
        output = norm_layer(input_tensor)
        
        # Vérifier la sortie
        expected = (input_tensor - mean) / std
        assert torch.allclose(output, expected, atol=1e-3)
    
    def test_invalid_method(self):
        """Tester qu'une erreur est levée pour une méthode invalide."""
        with pytest.raises(ValueError):
            NormalizationLayer(method='invalid_method', stats={})
    
    def test_missing_stats(self):
        """Tester qu'une erreur est levée pour des statistiques manquantes."""
        with pytest.raises(ValueError):
            NormalizationLayer(method='min-max', stats={'min': 0.0})  # max manquant
    
# ===================================================================================================
# Tests pour la fonction d'application de normalisation
# ===================================================================================================

def test_apply_normalization():
    """Tester l'application de la normalisation à un tenseur."""
    # Créer des données de test
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    
    # Appliquer la normalisation min-max
    output = apply_normalization(
        input_tensor,
        method='min-max',
        stats={'min': 1.0, 'max': 4.0}
    )
    
    # Vérifier la sortie
    expected = torch.tensor([[0.0, 1/3], [2/3, 1.0]])
    assert torch.allclose(output, expected, atol=1e-6)
    
    # Appliquer la normalisation z-score
    output = apply_normalization(
        input_tensor,
        method='z-score',
        stats={'mean': 2.5, 'std': 1.118}
    )
    
    # Vérifier la sortie
    expected = (input_tensor - 2.5) / 1.118
    assert torch.allclose(output, expected, atol=1e-3) 