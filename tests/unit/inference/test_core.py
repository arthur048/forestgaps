"""
Tests unitaires pour le module core.py de l'inférence.
"""

import os
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from forestgaps_dl.inference.core import InferenceConfig, InferenceResult, InferenceManager


class TestInferenceConfig:
    """Tests pour la classe InferenceConfig."""

    def test_default_values(self):
        """Vérifier que les valeurs par défaut sont correctement initialisées."""
        config = InferenceConfig()
        
        assert config.tiled_processing is False
        assert config.tile_size == 512
        assert config.tile_overlap == 64
        assert config.batch_size == 4
        assert config.threshold_probability == 0.5
        assert isinstance(config.post_processing, dict)
        
    def test_custom_values(self):
        """Vérifier que les valeurs personnalisées sont correctement initialisées."""
        custom_post_processing = {
            "apply_smoothing": True,
            "smoothing_kernel_size": 5,
            "remove_small_objects": True,
            "min_size": 200
        }
        
        config = InferenceConfig(
            tiled_processing=True,
            tile_size=1024,
            tile_overlap=128,
            batch_size=8,
            threshold_probability=0.7,
            post_processing=custom_post_processing
        )
        
        assert config.tiled_processing is True
        assert config.tile_size == 1024
        assert config.tile_overlap == 128
        assert config.batch_size == 8
        assert config.threshold_probability == 0.7
        assert config.post_processing == custom_post_processing
        
    def test_to_dict(self):
        """Vérifier que la méthode to_dict fonctionne correctement."""
        config = InferenceConfig(
            tiled_processing=True,
            tile_size=1024,
            batch_size=8
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["tiled_processing"] is True
        assert config_dict["tile_size"] == 1024
        assert config_dict["batch_size"] == 8


class TestInferenceResult:
    """Tests pour la classe InferenceResult."""
    
    @pytest.fixture
    def sample_result(self):
        """Créer un exemple de résultat d'inférence pour les tests."""
        predictions = np.zeros((100, 100), dtype=np.float32)
        predictions[40:60, 40:60] = 1.0  # Simuler une prédiction
        
        metadata = {
            "transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "crs": "EPSG:32631",
            "width": 100,
            "height": 100
        }
        
        return InferenceResult(
            predictions=predictions,
            dsm_path="test/path/dsm.tif",
            metadata=metadata,
            threshold=5.0
        )
    
    def test_initialization(self, sample_result):
        """Vérifier que l'initialisation fonctionne correctement."""
        assert sample_result.predictions.shape == (100, 100)
        assert sample_result.dsm_path == "test/path/dsm.tif"
        assert sample_result.threshold == 5.0
        assert "transform" in sample_result.metadata
        
    def test_binary_mask(self, sample_result):
        """Vérifier que le masque binaire est correctement généré."""
        binary_mask = sample_result.get_binary_mask()
        
        assert binary_mask.shape == (100, 100)
        assert binary_mask.dtype == np.bool_
        assert np.sum(binary_mask) == 400  # 20x20 pixels à 1.0
        
    @patch("forestgaps_dl.inference.core.save_raster")
    def test_save(self, mock_save_raster, sample_result):
        """Vérifier que la méthode save fonctionne correctement."""
        output_path = "test/output/prediction.tif"
        sample_result.save(output_path)
        
        mock_save_raster.assert_called_once()
        args, kwargs = mock_save_raster.call_args
        assert args[0] == sample_result.predictions
        assert args[1] == output_path
        assert kwargs["metadata"] == sample_result.metadata
        
    @patch("forestgaps_dl.inference.core.visualize_predictions")
    def test_visualize(self, mock_visualize, sample_result):
        """Vérifier que la méthode visualize fonctionne correctement."""
        sample_result.visualize()
        
        mock_visualize.assert_called_once()
        args, kwargs = mock_visualize.call_args
        assert np.array_equal(args[0], sample_result.predictions)


class TestInferenceManager:
    """Tests pour la classe InferenceManager."""
    
    @pytest.fixture
    def mock_model(self):
        """Créer un modèle simulé pour les tests."""
        model = MagicMock()
        model.eval.return_value = model
        model.to.return_value = model
        
        # Simuler une prédiction
        def forward_mock(x):
            batch_size = x.shape[0]
            return torch.ones((batch_size, 1, 64, 64))
        
        model.forward.side_effect = forward_mock
        return model
    
    @pytest.fixture
    def inference_manager(self, mock_model):
        """Créer un gestionnaire d'inférence pour les tests."""
        with patch("forestgaps_dl.inference.core.torch.load", return_value=mock_model):
            manager = InferenceManager(
                model_path="test/model.pt",
                config=InferenceConfig(),
                device="cpu"
            )
            return manager
    
    def test_initialization(self, inference_manager, mock_model):
        """Vérifier que l'initialisation fonctionne correctement."""
        assert inference_manager.model_path == "test/model.pt"
        assert inference_manager.device == "cpu"
        assert isinstance(inference_manager.config, InferenceConfig)
        assert inference_manager.model is not None
        
    @patch("forestgaps_dl.inference.core.load_raster")
    @patch("forestgaps_dl.inference.core.preprocess_dsm")
    @patch("forestgaps_dl.inference.core.postprocess_prediction")
    def test_predict(self, mock_postprocess, mock_preprocess, mock_load_raster, 
                    inference_manager, mock_model):
        """Vérifier que la méthode predict fonctionne correctement."""
        # Configurer les mocks
        mock_load_raster.return_value = (np.zeros((100, 100)), {"transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]})
        mock_preprocess.return_value = torch.zeros((1, 1, 100, 100))
        mock_postprocess.return_value = np.ones((100, 100))
        
        # Exécuter la prédiction
        result = inference_manager.predict(
            dsm_path="test/dsm.tif",
            threshold=5.0,
            output_path=None,
            visualize=False
        )
        
        # Vérifier les résultats
        assert isinstance(result, InferenceResult)
        assert result.dsm_path == "test/dsm.tif"
        assert result.threshold == 5.0
        
        # Vérifier que les mocks ont été appelés
        mock_load_raster.assert_called_once_with("test/dsm.tif")
        mock_preprocess.assert_called_once()
        mock_postprocess.assert_called_once()
        
    @patch("forestgaps_dl.inference.core.InferenceManager.predict")
    def test_predict_batch(self, mock_predict, inference_manager):
        """Vérifier que la méthode predict_batch fonctionne correctement."""
        # Configurer le mock
        mock_result = MagicMock()
        mock_predict.return_value = mock_result
        
        # Exécuter la prédiction par lots
        dsm_paths = ["test/dsm1.tif", "test/dsm2.tif"]
        results = inference_manager.predict_batch(
            dsm_paths=dsm_paths,
            threshold=5.0,
            output_dir="test/output",
            visualize=False
        )
        
        # Vérifier les résultats
        assert isinstance(results, dict)
        assert len(results) == 2
        assert "test/dsm1.tif" in results
        assert "test/dsm2.tif" in results
        
        # Vérifier que predict a été appelé pour chaque fichier
        assert mock_predict.call_count == 2 