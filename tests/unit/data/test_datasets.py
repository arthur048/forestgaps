"""
Tests unitaires pour le module de datasets.

Ce module contient des tests pour les datasets PyTorch qui chargent
et préparent les données pour l'entraînement et l'évaluation.
"""

import os
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from forestgaps_dl.data.datasets import (
    GapDataset,
    GapRegressionDataset,
    TarArchiveDataset
)

# ===================================================================================================
# Fixtures partagées
# ===================================================================================================

@pytest.fixture
def mock_tile_info():
    """Créer des informations sur les tuiles simulées pour les tests."""
    return {
        "site1_r0_c0": {
            "path": "site1_r0_c0_dsm.npy",
            "masks": {
                "2.0": "site1_r0_c0_mask_2.0.npy",
                "5.0": "site1_r0_c0_mask_5.0.npy",
                "10.0": "site1_r0_c0_mask_10.0.npy"
            },
            "metadata": {
                "valid_ratio": 0.95,
                "stats": {
                    "gap_ratio": {"2.0": 0.15, "5.0": 0.08, "10.0": 0.03}
                }
            }
        },
        "site1_r0_c1": {
            "path": "site1_r0_c1_dsm.npy",
            "masks": {
                "2.0": "site1_r0_c1_mask_2.0.npy",
                "5.0": "site1_r0_c1_mask_5.0.npy",
                "10.0": "site1_r0_c1_mask_10.0.npy"
            },
            "metadata": {
                "valid_ratio": 0.92,
                "stats": {
                    "gap_ratio": {"2.0": 0.18, "5.0": 0.09, "10.0": 0.04}
                }
            }
        }
    }

@pytest.fixture
def mock_dsm_data():
    """Créer des données DSM simulées."""
    return np.random.rand(64, 64).astype(np.float32)

@pytest.fixture
def mock_mask_data():
    """Créer des données de masque simulées."""
    return np.random.randint(0, 2, size=(64, 64)).astype(np.float32)

# ===================================================================================================
# Tests pour GapDataset
# ===================================================================================================

class TestGapDataset:
    """Tests pour la classe GapDataset."""
    
    @patch('numpy.load')
    def test_initialization(self, mock_load, mock_tile_info):
        """Tester l'initialisation du dataset."""
        # Configurer le dataset
        thresholds = [2.0, 5.0, 10.0]
        dataset = GapDataset(mock_tile_info, thresholds)
        
        # Vérifier les attributs
        assert len(dataset) == 2
        assert dataset.thresholds == thresholds
        assert dataset.tile_ids == ["site1_r0_c0", "site1_r0_c1"]
    
    @patch('numpy.load')
    def test_getitem(self, mock_load, mock_tile_info, mock_dsm_data, mock_mask_data):
        """Tester la récupération d'items du dataset."""
        # Configurer le mock pour charger les données simulées
        mock_load.side_effect = [mock_dsm_data] + [mock_mask_data] * 3  # 1 DSM, 3 masks
        
        # Configurer le dataset
        thresholds = [2.0, 5.0, 10.0]
        dataset = GapDataset(mock_tile_info, thresholds)
        
        # Récupérer un élément
        dsm, masks = dataset[0]
        
        # Vérifier la forme des données
        assert isinstance(dsm, torch.Tensor)
        assert isinstance(masks, list)
        assert len(masks) == 3  # Un masque par seuil
        assert dsm.shape == (1, 64, 64)  # Canal ajouté
        assert masks[0].shape == (1, 64, 64)  # Canal ajouté
    
    @patch('numpy.load')
    def test_normalization(self, mock_load, mock_tile_info, mock_dsm_data, mock_mask_data):
        """Tester la normalisation des données."""
        # Configurer le mock
        mock_load.side_effect = [mock_dsm_data] + [mock_mask_data] * 3
        
        # Créer des statistiques de normalisation
        norm_stats = {
            'min': 0.0,
            'max': 1.0,
            'mean': 0.5,
            'std': 0.2
        }
        
        # Configurer le dataset avec normalisation min-max
        dataset = GapDataset(
            mock_tile_info,
            [2.0, 5.0, 10.0],
            normalization_stats=norm_stats,
            normalization_method='min-max'
        )
        
        # Récupérer un élément
        dsm, _ = dataset[0]
        
        # Vérifier que les valeurs sont normalisées (entre 0 et 1)
        assert torch.min(dsm) >= 0.0
        assert torch.max(dsm) <= 1.0

# ===================================================================================================
# Tests pour GapRegressionDataset
# ===================================================================================================

class TestGapRegressionDataset:
    """Tests pour la classe GapRegressionDataset."""
    
    @patch('numpy.load')
    def test_initialization(self, mock_load, mock_tile_info):
        """Tester l'initialisation du dataset de régression."""
        # Configurer le dataset
        dataset = GapRegressionDataset(mock_tile_info)
        
        # Vérifier les attributs
        assert len(dataset) == 2
        assert dataset.tile_ids == ["site1_r0_c0", "site1_r0_c1"]
    
    @patch('numpy.load')
    def test_getitem(self, mock_load, mock_tile_info, mock_dsm_data, mock_mask_data):
        """Tester la récupération d'items du dataset de régression."""
        # Configurer le mock
        mock_chm_data = mock_dsm_data - 2.0  # Simuler des données CHM
        mock_load.side_effect = [mock_dsm_data, mock_chm_data]
        
        # Configurer le dataset
        dataset = GapRegressionDataset(mock_tile_info)
        
        # Récupérer un élément
        dsm, chm = dataset[0]
        
        # Vérifier la forme des données
        assert isinstance(dsm, torch.Tensor)
        assert isinstance(chm, torch.Tensor)
        assert dsm.shape == (1, 64, 64)
        assert chm.shape == (1, 64, 64)

# ===================================================================================================
# Tests pour TarArchiveDataset
# ===================================================================================================

class TestTarArchiveDataset:
    """Tests pour la classe TarArchiveDataset."""
    
    @patch('tarfile.open')
    @patch('json.load')
    def test_initialization(self, mock_json_load, mock_tarfile_open, mock_tile_info):
        """Tester l'initialisation du dataset d'archive tar."""
        # Configurer les mocks
        mock_tarfile = MagicMock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tarfile
        mock_json_load.return_value = mock_tile_info
        
        # Configurer le dataset
        dataset = TarArchiveDataset("dummy_archive.tar", "dummy_index.json")
        
        # Vérifier les attributs
        assert dataset.tile_ids == ["site1_r0_c0", "site1_r0_c1"]
    
    @patch('tarfile.open')
    @patch('json.load')
    @patch('io.BytesIO')
    @patch('numpy.load')
    def test_getitem(self, mock_np_load, mock_bytesio, mock_json_load, 
                    mock_tarfile_open, mock_tile_info, mock_dsm_data, mock_mask_data):
        """Tester la récupération d'items du dataset d'archive tar."""
        # Configurer les mocks
        mock_tarfile = MagicMock()
        mock_tarfile_open.return_value.__enter__.return_value = mock_tarfile
        mock_json_load.return_value = mock_tile_info
        
        # Simuler l'extraction de fichiers de l'archive
        mock_bytesio.return_value = MagicMock()
        mock_np_load.side_effect = [mock_dsm_data] + [mock_mask_data] * 3
        
        # Configurer le dataset
        dataset = TarArchiveDataset(
            "dummy_archive.tar",
            "dummy_index.json",
            thresholds=[2.0, 5.0, 10.0]
        )
        
        # Récupérer un élément
        dsm, masks = dataset[0]
        
        # Vérifier la forme des données
        assert isinstance(dsm, torch.Tensor)
        assert isinstance(masks, list)
        assert len(masks) == 3
        assert dsm.shape == (1, 64, 64)
        assert masks[0].shape == (1, 64, 64) 