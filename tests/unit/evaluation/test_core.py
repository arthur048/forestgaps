"""
Tests unitaires pour le module core.py de l'évaluation.
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock

from forestgaps_dl.evaluation.core import EvaluationConfig, EvaluationResult, ExternalEvaluator


class TestEvaluationConfig:
    """Tests pour la classe EvaluationConfig."""

    def test_default_values(self):
        """Vérifier que les valeurs par défaut sont correctement initialisées."""
        config = EvaluationConfig()
        
        assert config.thresholds == [2.0, 5.0, 10.0, 15.0]
        assert config.compare_with_previous is False
        assert config.previous_model_path is None
        assert config.save_predictions is True
        assert config.save_visualizations is True
        assert config.detailed_reporting is True
        assert isinstance(config.metrics, list)
        assert config.batch_size == 4
        assert config.num_workers == 4
        assert config.tiled_processing is False
        
    def test_custom_values(self):
        """Vérifier que les valeurs personnalisées sont correctement initialisées."""
        custom_metrics = ["precision", "recall", "f1"]
        
        config = EvaluationConfig(
            thresholds=[3.0, 6.0, 9.0],
            compare_with_previous=True,
            previous_model_path="models/previous.pt",
            save_predictions=False,
            save_visualizations=False,
            detailed_reporting=False,
            metrics=custom_metrics,
            batch_size=8,
            num_workers=2,
            tiled_processing=True
        )
        
        assert config.thresholds == [3.0, 6.0, 9.0]
        assert config.compare_with_previous is True
        assert config.previous_model_path == "models/previous.pt"
        assert config.save_predictions is False
        assert config.save_visualizations is False
        assert config.detailed_reporting is False
        assert config.metrics == custom_metrics
        assert config.batch_size == 8
        assert config.num_workers == 2
        assert config.tiled_processing is True
        
    def test_to_dict(self):
        """Vérifier que la méthode to_dict fonctionne correctement."""
        config = EvaluationConfig(
            thresholds=[3.0, 6.0, 9.0],
            batch_size=8,
            tiled_processing=True
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["thresholds"] == [3.0, 6.0, 9.0]
        assert config_dict["batch_size"] == 8
        assert config_dict["tiled_processing"] is True


class TestEvaluationResult:
    """Tests pour la classe EvaluationResult."""
    
    @pytest.fixture
    def sample_metrics(self):
        """Créer un exemple de métriques pour les tests."""
        metrics = {
            "overall": {
                "precision": 0.85,
                "recall": 0.78,
                "f1": 0.81,
                "iou": 0.72
            },
            "by_threshold": {
                2.0: {"precision": 0.82, "recall": 0.75, "f1": 0.78, "iou": 0.68},
                5.0: {"precision": 0.87, "recall": 0.80, "f1": 0.83, "iou": 0.74},
                10.0: {"precision": 0.90, "recall": 0.85, "f1": 0.87, "iou": 0.78}
            }
        }
        return metrics
    
    @pytest.fixture
    def sample_result(self, sample_metrics):
        """Créer un exemple de résultat d'évaluation pour les tests."""
        predictions = np.zeros((100, 100), dtype=np.float32)
        predictions[40:60, 40:60] = 1.0  # Simuler une prédiction
        
        ground_truth = np.zeros((100, 100), dtype=np.float32)
        ground_truth[38:62, 38:62] = 1.0  # Simuler une vérité terrain
        
        return EvaluationResult(
            metrics=sample_metrics,
            predictions=predictions,
            ground_truth=ground_truth,
            dsm_path="test/path/dsm.tif",
            chm_path="test/path/chm.tif",
            thresholds=[2.0, 5.0, 10.0],
            model_name="unet_test"
        )
    
    def test_initialization(self, sample_result, sample_metrics):
        """Vérifier que l'initialisation fonctionne correctement."""
        assert sample_result.metrics == sample_metrics
        assert sample_result.predictions.shape == (100, 100)
        assert sample_result.ground_truth.shape == (100, 100)
        assert sample_result.dsm_path == "test/path/dsm.tif"
        assert sample_result.chm_path == "test/path/chm.tif"
        assert sample_result.thresholds == [2.0, 5.0, 10.0]
        assert sample_result.model_name == "unet_test"
        
    def test_get_metrics_dataframe(self, sample_result):
        """Vérifier que la méthode get_metrics_dataframe fonctionne correctement."""
        df = sample_result.get_metrics_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert "threshold" in df.columns
        assert "precision" in df.columns
        assert "recall" in df.columns
        assert "f1" in df.columns
        assert "iou" in df.columns
        assert len(df) == 4  # 3 thresholds + overall
        
    @patch("forestgaps_dl.evaluation.core.save_metrics_to_csv")
    def test_save_metrics(self, mock_save_metrics, sample_result):
        """Vérifier que la méthode save_metrics fonctionne correctement."""
        output_path = "test/output/metrics.csv"
        sample_result.save_metrics(output_path)
        
        mock_save_metrics.assert_called_once()
        args, kwargs = mock_save_metrics.call_args
        assert args[0] == sample_result.get_metrics_dataframe()
        assert args[1] == output_path
        
    @patch("forestgaps_dl.evaluation.core.visualize_metrics")
    def test_visualize(self, mock_visualize, sample_result):
        """Vérifier que la méthode visualize fonctionne correctement."""
        sample_result.visualize()
        
        mock_visualize.assert_called_once()
        args, kwargs = mock_visualize.call_args
        assert args[0] == sample_result.predictions
        assert args[1] == sample_result.ground_truth
        
    @patch("forestgaps_dl.evaluation.core.generate_evaluation_report")
    def test_generate_report(self, mock_generate_report, sample_result):
        """Vérifier que la méthode generate_report fonctionne correctement."""
        output_dir = "test/output"
        sample_result.generate_report(output_dir)
        
        mock_generate_report.assert_called_once()
        args, kwargs = mock_generate_report.call_args
        assert args[0] == sample_result
        assert args[1] == output_dir


class TestExternalEvaluator:
    """Tests pour la classe ExternalEvaluator."""
    
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
    def external_evaluator(self, mock_model):
        """Créer un évaluateur externe pour les tests."""
        with patch("forestgaps_dl.evaluation.core.torch.load", return_value=mock_model):
            evaluator = ExternalEvaluator(
                model_path="test/model.pt",
                config=EvaluationConfig(),
                device="cpu"
            )
            return evaluator
    
    def test_initialization(self, external_evaluator, mock_model):
        """Vérifier que l'initialisation fonctionne correctement."""
        assert external_evaluator.model_path == "test/model.pt"
        assert external_evaluator.device == "cpu"
        assert isinstance(external_evaluator.config, EvaluationConfig)
        assert external_evaluator.model is not None
        
    @patch("forestgaps_dl.evaluation.core.load_raster")
    @patch("forestgaps_dl.evaluation.core.calculate_metrics")
    def test_create_ground_truth(self, mock_calculate_metrics, mock_load_raster, external_evaluator):
        """Vérifier que la méthode _create_ground_truth fonctionne correctement."""
        # Configurer les mocks
        chm_data = np.zeros((100, 100))
        chm_data[40:60, 40:60] = 8.0  # Simuler une zone de trouée
        mock_load_raster.return_value = (chm_data, {"transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]})
        
        # Exécuter la création de vérité terrain
        thresholds = [2.0, 5.0, 10.0]
        ground_truth = external_evaluator._create_ground_truth("test/chm.tif", thresholds)
        
        # Vérifier les résultats
        assert isinstance(ground_truth, dict)
        assert len(ground_truth) == len(thresholds)
        assert 2.0 in ground_truth
        assert 5.0 in ground_truth
        assert 10.0 in ground_truth
        
        # Vérifier que les mocks ont été appelés
        mock_load_raster.assert_called_once_with("test/chm.tif")
        
    @patch("forestgaps_dl.evaluation.core.ExternalEvaluator._create_ground_truth")
    @patch("forestgaps_dl.evaluation.core.InferenceManager")
    @patch("forestgaps_dl.evaluation.core.calculate_metrics")
    def test_evaluate(self, mock_calculate_metrics, mock_inference_manager, 
                     mock_create_ground_truth, external_evaluator):
        """Vérifier que la méthode evaluate fonctionne correctement."""
        # Configurer les mocks
        mock_ground_truth = {
            2.0: np.ones((100, 100), dtype=bool),
            5.0: np.ones((100, 100), dtype=bool),
            10.0: np.ones((100, 100), dtype=bool)
        }
        mock_create_ground_truth.return_value = mock_ground_truth
        
        mock_inference_result = MagicMock()
        mock_inference_result.predictions = np.ones((100, 100))
        mock_inference_manager_instance = MagicMock()
        mock_inference_manager_instance.predict.return_value = mock_inference_result
        mock_inference_manager.return_value = mock_inference_manager_instance
        
        mock_calculate_metrics.return_value = {"precision": 0.9, "recall": 0.8, "f1": 0.85, "iou": 0.75}
        
        # Exécuter l'évaluation
        result = external_evaluator.evaluate(
            dsm_path="test/dsm.tif",
            chm_path="test/chm.tif",
            thresholds=[2.0, 5.0, 10.0],
            output_dir="test/output",
            visualize=False
        )
        
        # Vérifier les résultats
        assert isinstance(result, EvaluationResult)
        assert result.dsm_path == "test/dsm.tif"
        assert result.chm_path == "test/chm.tif"
        
        # Vérifier que les mocks ont été appelés
        mock_create_ground_truth.assert_called_once_with("test/chm.tif", [2.0, 5.0, 10.0])
        mock_inference_manager_instance.predict.assert_called_once()
        assert mock_calculate_metrics.call_count >= 1
        
    @patch("forestgaps_dl.evaluation.core.ExternalEvaluator.evaluate")
    @patch("forestgaps_dl.evaluation.core.glob.glob")
    def test_evaluate_site(self, mock_glob, mock_evaluate, external_evaluator):
        """Vérifier que la méthode evaluate_site fonctionne correctement."""
        # Configurer les mocks
        mock_glob.side_effect = [
            ["test/site/dsm1.tif", "test/site/dsm2.tif"],  # DSM files
            ["test/site/chm1.tif", "test/site/chm2.tif"]   # CHM files
        ]
        
        mock_result = MagicMock()
        mock_evaluate.return_value = mock_result
        mock_result.metrics = {"overall": {"precision": 0.9}}
        
        # Exécuter l'évaluation du site
        result = external_evaluator.evaluate_site(
            site_dsm_dir="test/site/dsm",
            site_chm_dir="test/site/chm",
            thresholds=[2.0, 5.0, 10.0],
            output_dir="test/output",
            visualize=False
        )
        
        # Vérifier les résultats
        assert isinstance(result, EvaluationResult)
        
        # Vérifier que les mocks ont été appelés
        assert mock_glob.call_count == 2
        assert mock_evaluate.call_count == 2 