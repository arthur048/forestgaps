"""
Tests d'intégration pour le flux de données.

Ce module contient des tests qui valident l'intégration entre
les composants de chargement de données et les modèles.
"""

import os
import tempfile
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from forestgaps.config import forestgaps.configurationManager
from forestgaps.data.datasets import GapDataset
from forestgaps.data.loaders import create_data_loaders
from forestgaps.models import create_model, ModelRegistry
from forestgaps.models.unet import UNet

# ===================================================================================================
# Fixtures pour les tests d'intégration
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
def mock_config():
    """Créer une configuration simulée pour les tests."""
    config_data = {
        "data": {
            "tile_size": 64,
            "overlap": 0.2,
            "gap_thresholds": [2.0, 5.0, 10.0],
            "batch_size": 4,
            "num_workers": 1,
            "normalization_method": "min-max"
        },
        "model": {
            "type": "unet",
            "params": {
                "in_channels": 1,
                "dropout_rate": 0.2
            }
        },
        "training": {
            "epochs": 5,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "batch_size": 4
        }
    }
    return ConfigurationManager(config_data)

@pytest.fixture
def mock_dataset_factory(mock_tile_info):
    """Créer une factory pour générer des datasets simulés."""
    
    def _create_dataset(thresholds=None):
        """Créer un dataset simulé."""
        if thresholds is None:
            thresholds = [2.0, 5.0, 10.0]
            
        # Simuler les données DSM et les masques
        dsm_data = np.random.rand(64, 64).astype(np.float32)
        mask_data = np.random.randint(0, 2, size=(64, 64)).astype(np.float32)
        
        # Patcher numpy.load pour retourner les données simulées
        with patch('numpy.load') as mock_load:
            mock_load.side_effect = [dsm_data] + [mask_data] * len(thresholds)
            
            # Créer le dataset
            dataset = GapDataset(
                tile_info=mock_tile_info,
                thresholds=thresholds,
                normalization_method="min-max",
                normalization_stats={
                    'min': 0.0,
                    'max': 1.0,
                    'mean': 0.5,
                    'std': 0.2
                }
            )
            
            return dataset
    
    return _create_dataset

# ===================================================================================================
# Tests d'intégration Dataset -> DataLoader -> Model
# ===================================================================================================

class TestDataToModel:
    """Tests d'intégration du flux de données vers le modèle."""
    
    @patch('torch.utils.data.DataLoader')
    def test_dataloader_creation(self, mock_dataloader, mock_dataset_factory, mock_config):
        """Tester la création de DataLoaders à partir d'un Dataset."""
        # Créer un dataset simulé
        dataset = mock_dataset_factory()
        
        # Patcher la méthode GapDataset pour retourner notre dataset simulé
        with patch('forestgaps.data.loaders.GapDataset', return_value=dataset):
            # Créer les dataloaders
            dataloaders = create_data_loaders(
                config=mock_config,
                batch_size=mock_config.get("training.batch_size"),
                num_workers=mock_config.get("data.num_workers")
            )
            
            # Vérifier que les dataloaders ont été créés
            assert "train" in dataloaders
            assert "val" in dataloaders
            assert "test" in dataloaders
    
    @patch('forestgaps.data.datasets.GapDataset.__getitem__')
    def test_model_forward_pass(self, mock_getitem, mock_config):
        """Tester l'intégration du modèle avec les données du Dataset."""
        # Enregistrer un modèle UNet de test
        ModelRegistry.register("unet", UNet)
        
        # Créer un modèle
        model = create_model(
            model_type="unet",
            **mock_config.get("model.params", {})
        )
        
        # Préparer des données simulées pour le modèle
        dsm = torch.randn(4, 1, 64, 64)  # [B, C, H, W]
        threshold = torch.tensor([5.0])  # La valeur de seuil
        
        # Définir le modèle en mode évaluation
        model.eval()
        
        # Exécuter une passe avant (inference)
        with torch.no_grad():
            output = model(dsm, threshold)
        
        # Vérifier la forme de la sortie
        assert output.shape == (4, 1, 64, 64)  # [B, C, H, W]
    
    def test_end_to_end_inference(self, mock_dataset_factory, mock_config):
        """Tester le flux complet des données au modèle et jusqu'à l'inférence."""
        # Créer un dataset simulé
        dataset = mock_dataset_factory()
        
        # Créer un DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=False
        )
        
        # Enregistrer un modèle UNet de test
        ModelRegistry.register("unet", UNet)
        
        # Créer un modèle
        model = create_model(
            model_type="unet",
            **mock_config.get("model.params", {})
        )
        
        # Définir le modèle en mode évaluation
        model.eval()
        
        # Charger un lot de données
        for dsm, masks in dataloader:
            # Préparer le seuil
            threshold = torch.tensor([5.0])
            
            # Exécuter une passe avant
            with torch.no_grad():
                output = model(dsm, threshold)
            
            # Vérifier la forme de la sortie
            assert output.shape == dsm.shape
            
            # Vérifier que la sortie contient des valeurs dans [0, 1]
            assert torch.all(output >= 0.0)
            assert torch.all(output <= 1.0)
            
            # Ne tester qu'un seul lot
            break 