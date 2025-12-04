"""
Suite de tests complète pour ForestGaps.

Tests de régression pour valider que tout fonctionne après modifications.
Run avec: pytest tests/test_complete_workflow.py -v
"""

import pytest
import torch
import numpy as np
from pathlib import Path


class TestModelRegistry:
    """Tests du registre de modèles."""

    def test_registry_has_all_models(self):
        """Vérifier que les 9 modèles sont disponibles."""
        from forestgaps.models import model_registry

        models = model_registry.list_models()
        assert len(models) == 9, f"Expected 9 models, got {len(models)}"

        expected_models = [
            'unet', 'attention_unet', 'resunet', 'film_unet',
            'unet_all_features', 'deeplabv3_plus',
            'deeplabv3_plus_threshold', 'regression_unet',
            'regression_unet_threshold'
        ]
        for model_name in expected_models:
            assert model_name in models, f"Model {model_name} not found in registry"

    def test_create_unet(self):
        """Tester création modèle U-Net."""
        from forestgaps.models import model_registry

        model = model_registry.create('unet', in_channels=1, out_channels=1)
        assert model is not None
        assert hasattr(model, 'forward')
        assert model.get_num_parameters() > 0

    def test_create_deeplabv3(self):
        """Tester création DeepLabV3+."""
        from forestgaps.models import model_registry

        model = model_registry.create('deeplabv3_plus', in_channels=1, out_channels=1)
        assert model is not None
        assert hasattr(model, 'get_complexity'), "DeepLabV3+ missing get_complexity()"

    def test_all_models_instantiate(self):
        """Tester que TOUS les modèles s'instancient."""
        from forestgaps.models import model_registry

        results = {}
        for model_name in model_registry.list_models():
            try:
                if 'regression' in model_name:
                    params = {'in_channels': 1, 'out_channels': 1}
                elif 'threshold' in model_name:
                    params = {'in_channels': 1, 'out_channels': 1, 'threshold_value': 5.0}
                else:
                    params = {'in_channels': 1, 'out_channels': 1}

                model = model_registry.create(model_name, **params)
                results[model_name] = 'OK'
            except Exception as e:
                results[model_name] = f'FAILED: {str(e)}'

        failures = {k: v for k, v in results.items() if v != 'OK'}
        assert len(failures) == 0, f"Models failed to instantiate: {failures}"


class TestModulesImport:
    """Tests des imports de modules."""

    def test_import_forestgaps(self):
        """Tester import package principal."""
        import forestgaps
        assert hasattr(forestgaps, '__version__')

    def test_import_inference(self):
        """Tester import module inference."""
        from forestgaps.inference import InferenceManager
        assert InferenceManager is not None

    def test_import_evaluation(self):
        """Tester import module evaluation."""
        from forestgaps.evaluation import evaluate_model, ExternalEvaluator
        assert evaluate_model is not None
        assert ExternalEvaluator is not None

    def test_import_training(self):
        """Tester import module training."""
        from forestgaps.training import Trainer
        assert Trainer is not None

    def test_import_benchmarking(self):
        """Tester import module benchmarking."""
        from forestgaps.benchmarking import ModelComparison
        assert ModelComparison is not None


class TestInferenceModule:
    """Tests du module inference."""

    def test_preprocess_dsm(self):
        """Tester preprocessing DSM."""
        from forestgaps.inference.utils.processing import preprocess_dsm

        data = np.random.randn(256, 256).astype(np.float32)
        result = preprocess_dsm(data, method="min-max")

        assert result.shape == data.shape
        assert not np.isnan(result).any()

    def test_postprocess_prediction(self):
        """Tester postprocessing."""
        from forestgaps.inference.utils.processing import postprocess_prediction

        prediction = (np.random.rand(256, 256) > 0.5).astype(np.uint8)
        result = postprocess_prediction(prediction, method="morphology")

        assert result.shape == prediction.shape

    def test_batch_predict(self):
        """Tester batch inference."""
        from forestgaps.inference.utils.processing import batch_predict
        from forestgaps.models import model_registry

        model = model_registry.create('unet', in_channels=1, out_channels=1)
        model.eval()

        batch = torch.randn(2, 1, 256, 256)
        predictions = batch_predict(model, batch, device='cpu')

        assert predictions.shape == (2, 1, 256, 256)


class TestEvaluationModule:
    """Tests du module evaluation."""

    def test_metrics_import(self):
        """Tester import metrics."""
        from forestgaps.evaluation.utils.metrics import (
            calculate_metrics,
            calculate_confusion_matrix,
            calculate_threshold_metrics
        )
        assert calculate_metrics is not None

    def test_visualization_import(self):
        """Tester import visualization."""
        from forestgaps.evaluation.utils.visualization import (
            visualize_metrics,
            visualize_comparison,
            create_metrics_table
        )
        assert visualize_metrics is not None

    def test_reporting_import(self):
        """Tester import reporting."""
        from forestgaps.evaluation.utils.reporting import (
            generate_evaluation_report,
            save_metrics_to_csv,
            create_site_comparison
        )
        assert generate_evaluation_report is not None


class TestModelForward:
    """Tests des forward passes des modèles."""

    @pytest.mark.parametrize("model_name", [
        'unet', 'attention_unet', 'resunet', 'film_unet', 'unet_all_features'
    ])
    def test_segmentation_models_forward(self, model_name):
        """Tester forward pass modèles segmentation."""
        from forestgaps.models import model_registry

        model = model_registry.create(model_name, in_channels=1, out_channels=1)
        model.eval()

        x = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            y = model(x)

        assert y.shape == (2, 1, 256, 256), f"{model_name}: wrong output shape"

    def test_deeplabv3_forward(self):
        """Tester forward pass DeepLabV3+."""
        from forestgaps.models import model_registry

        model = model_registry.create('deeplabv3_plus', in_channels=1, out_channels=1)
        model.eval()

        x = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            y = model(x)

        assert y.shape == (2, 1, 256, 256)

    def test_regression_model_forward(self):
        """Tester forward pass modèle régression."""
        from forestgaps.models import model_registry

        model = model_registry.create('regression_unet', in_channels=1, out_channels=1)
        model.eval()

        x = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            y = model(x)

        assert y.shape == (2, 1, 256, 256)


class TestTraining:
    """Tests du training."""

    def test_simple_training_loop(self):
        """Tester loop training minimal."""
        from forestgaps.models import model_registry
        import torch.nn as nn
        import torch.optim as optim

        model = model_registry.create('unet', in_channels=1, out_channels=1)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Dummy data
        x = torch.randn(4, 1, 256, 256)
        y = torch.randint(0, 2, (4, 1, 256, 256)).float()

        # Forward
        model.train()
        output = model(x)
        loss = criterion(output, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0


class TestSaveLoad:
    """Tests sauvegarde/chargement modèles."""

    def test_save_load_model(self, tmp_path):
        """Tester sauvegarde et rechargement."""
        from forestgaps.models import model_registry

        # Créer modèle
        model = model_registry.create('unet', in_channels=1, out_channels=1)

        # Sauvegarder
        model_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), model_path)

        # Recharger
        new_model = model_registry.create('unet', in_channels=1, out_channels=1)
        new_model.load_state_dict(torch.load(model_path))

        # Vérifier
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            y1 = model(x)
            y2 = new_model(x)

        assert torch.allclose(y1, y2, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
