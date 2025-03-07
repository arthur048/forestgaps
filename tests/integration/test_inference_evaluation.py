"""
Tests d'intégration pour vérifier l'interaction entre les modules d'inférence et d'évaluation.
"""

import os
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from forestgaps_dl.inference import InferenceManager, InferenceConfig, run_inference
from forestgaps_dl.evaluation import ExternalEvaluator, EvaluationConfig, evaluate_model


class TestInferenceEvaluationIntegration:
    """Tests d'intégration pour les modules d'inférence et d'évaluation."""
    
    @pytest.fixture
    def mock_model(self):
        """Créer un modèle simulé pour les tests."""
        model = MagicMock()
        model.eval.return_value = model
        model.to.return_value = model
        
        # Simuler une prédiction
        def forward_mock(x):
            batch_size = x.shape[0]
            height, width = x.shape[2], x.shape[3]
            return torch.ones((batch_size, 1, height, width))
        
        model.forward.side_effect = forward_mock
        return model
    
    @pytest.fixture
    def mock_raster_data(self):
        """Créer des données raster simulées pour les tests."""
        # Créer un DSM simulé
        dsm = np.zeros((200, 200), dtype=np.float32)
        dsm[50:150, 50:150] = 30.0  # Zone élevée
        
        # Créer un CHM simulé
        chm = np.zeros((200, 200), dtype=np.float32)
        chm[50:150, 50:150] = 20.0  # Canopée
        chm[80:120, 80:120] = 2.0   # Trouée
        
        # Métadonnées géospatiales
        metadata = {
            "transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "crs": "EPSG:32631",
            "width": 200,
            "height": 200
        }
        
        return dsm, chm, metadata
    
    @patch("forestgaps_dl.inference.core.torch.load")
    @patch("forestgaps_dl.inference.core.load_raster")
    @patch("forestgaps_dl.inference.core.save_raster")
    @patch("forestgaps_dl.evaluation.core.load_raster")
    def test_inference_to_evaluation_workflow(self, mock_eval_load_raster, mock_save_raster, 
                                             mock_inf_load_raster, mock_torch_load, 
                                             mock_model, mock_raster_data, tmp_path):
        """
        Tester le flux de travail complet d'inférence à évaluation.
        
        Ce test simule le processus suivant :
        1. Exécution de l'inférence sur un DSM
        2. Sauvegarde des prédictions
        3. Évaluation des prédictions par rapport à un CHM
        """
        # Configurer les mocks
        dsm, chm, metadata = mock_raster_data
        mock_torch_load.return_value = mock_model
        
        # Configurer le mock pour load_raster dans le module d'inférence
        mock_inf_load_raster.return_value = (dsm, metadata)
        
        # Configurer le mock pour load_raster dans le module d'évaluation
        # Premier appel : charger le CHM
        # Deuxième appel : charger les prédictions sauvegardées
        mock_eval_load_raster.side_effect = [
            (chm, metadata),
            (np.ones((200, 200)), metadata)
        ]
        
        # Créer des chemins temporaires pour les tests
        dsm_path = os.path.join(tmp_path, "dsm.tif")
        chm_path = os.path.join(tmp_path, "chm.tif")
        output_path = os.path.join(tmp_path, "prediction.tif")
        output_dir = os.path.join(tmp_path, "evaluation")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Exécuter l'inférence
        inference_config = InferenceConfig(
            tiled_processing=True,
            tile_size=64,
            tile_overlap=8,
            batch_size=4
        )
        
        inference_manager = InferenceManager(
            model_path="test/model.pt",
            config=inference_config,
            device="cpu"
        )
        
        inference_result = inference_manager.predict(
            dsm_path=dsm_path,
            threshold=5.0,
            output_path=output_path,
            visualize=False
        )
        
        # Vérifier que l'inférence a produit un résultat
        assert inference_result is not None
        assert isinstance(inference_result.predictions, np.ndarray)
        assert inference_result.predictions.shape == (200, 200)
        
        # Vérifier que save_raster a été appelé pour sauvegarder les prédictions
        mock_save_raster.assert_called_once()
        
        # 2. Exécuter l'évaluation
        evaluation_config = EvaluationConfig(
            thresholds=[2.0, 5.0, 10.0],
            save_predictions=True,
            save_visualizations=True
        )
        
        evaluator = ExternalEvaluator(
            model_path="test/model.pt",
            config=evaluation_config,
            device="cpu"
        )
        
        # Remplacer la méthode predict de l'InferenceManager pour utiliser notre résultat simulé
        with patch("forestgaps_dl.evaluation.core.InferenceManager") as mock_inf_manager:
            mock_inf_manager_instance = MagicMock()
            mock_inf_manager_instance.predict.return_value = inference_result
            mock_inf_manager.return_value = mock_inf_manager_instance
            
            evaluation_result = evaluator.evaluate(
                dsm_path=dsm_path,
                chm_path=chm_path,
                thresholds=[2.0, 5.0, 10.0],
                output_dir=output_dir,
                visualize=False
            )
        
        # Vérifier que l'évaluation a produit un résultat
        assert evaluation_result is not None
        assert "overall" in evaluation_result.metrics
        assert "by_threshold" in evaluation_result.metrics
        assert 2.0 in evaluation_result.metrics["by_threshold"]
        assert 5.0 in evaluation_result.metrics["by_threshold"]
        assert 10.0 in evaluation_result.metrics["by_threshold"]
        
    @patch("forestgaps_dl.inference.torch.load")
    @patch("forestgaps_dl.inference.load_raster")
    @patch("forestgaps_dl.evaluation.torch.load")
    @patch("forestgaps_dl.evaluation.load_raster")
    def test_high_level_api_integration(self, mock_eval_load_raster, mock_eval_torch_load,
                                       mock_inf_load_raster, mock_inf_torch_load,
                                       mock_model, mock_raster_data, tmp_path):
        """
        Tester l'intégration des API de haut niveau pour l'inférence et l'évaluation.
        
        Ce test simule l'utilisation des fonctions de haut niveau :
        1. run_inference pour exécuter l'inférence
        2. evaluate_model pour évaluer les résultats
        """
        # Configurer les mocks
        dsm, chm, metadata = mock_raster_data
        mock_inf_torch_load.return_value = mock_model
        mock_eval_torch_load.return_value = mock_model
        
        # Configurer les mocks pour load_raster
        mock_inf_load_raster.return_value = (dsm, metadata)
        mock_eval_load_raster.side_effect = [
            (chm, metadata),
            (np.ones((200, 200)), metadata)
        ]
        
        # Créer des chemins temporaires pour les tests
        dsm_path = os.path.join(tmp_path, "dsm.tif")
        chm_path = os.path.join(tmp_path, "chm.tif")
        output_path = os.path.join(tmp_path, "prediction.tif")
        output_dir = os.path.join(tmp_path, "evaluation")
        os.makedirs(output_dir, exist_ok=True)
        
        # Remplacer les fonctions d'inférence et d'évaluation par des mocks
        with patch("forestgaps_dl.inference.InferenceManager") as mock_inf_manager, \
             patch("forestgaps_dl.evaluation.ExternalEvaluator") as mock_evaluator:
            
            # Configurer le mock pour InferenceManager
            mock_inf_result = MagicMock()
            mock_inf_result.predictions = np.ones((200, 200))
            mock_inf_manager_instance = MagicMock()
            mock_inf_manager_instance.predict.return_value = mock_inf_result
            mock_inf_manager.return_value = mock_inf_manager_instance
            
            # Configurer le mock pour ExternalEvaluator
            mock_eval_result = MagicMock()
            mock_eval_result.metrics = {
                "overall": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "iou": 0.75},
                "by_threshold": {
                    2.0: {"precision": 0.85, "recall": 0.75, "f1": 0.8, "iou": 0.7},
                    5.0: {"precision": 0.9, "recall": 0.8, "f1": 0.85, "iou": 0.75},
                    10.0: {"precision": 0.95, "recall": 0.85, "f1": 0.9, "iou": 0.8}
                }
            }
            mock_evaluator_instance = MagicMock()
            mock_evaluator_instance.evaluate.return_value = mock_eval_result
            mock_evaluator.return_value = mock_evaluator_instance
            
            # 1. Exécuter l'inférence avec l'API de haut niveau
            with patch("forestgaps_dl.inference.InferenceManager", return_value=mock_inf_manager_instance):
                inference_result = run_inference(
                    model_path="test/model.pt",
                    dsm_path=dsm_path,
                    output_path=output_path,
                    threshold=5.0,
                    visualize=False
                )
            
            # 2. Exécuter l'évaluation avec l'API de haut niveau
            with patch("forestgaps_dl.evaluation.ExternalEvaluator", return_value=mock_evaluator_instance):
                evaluation_result = evaluate_model(
                    model_path="test/model.pt",
                    dsm_path=dsm_path,
                    chm_path=chm_path,
                    output_dir=output_dir,
                    thresholds=[2.0, 5.0, 10.0],
                    visualize=False
                )
        
        # Vérifier que l'inférence a produit un résultat
        assert inference_result is not None
        
        # Vérifier que l'évaluation a produit un résultat
        assert evaluation_result is not None
        assert mock_evaluator_instance.evaluate.called 