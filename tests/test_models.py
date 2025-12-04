"""
Tests unitaires pour tous les modèles ForestGaps.

Vérifie que chaque modèle peut:
- S'instancier correctement
- Faire un forward pass
- Gérer les paramètres FiLM si applicable
"""

import pytest
import torch
from forestgaps.models import create_model


class TestModelCreation:
    """Test l'instanciation de tous les modèles."""

    @pytest.mark.parametrize("model_type,init_features", [
        ("unet", 32),
        ("unet", 64),
        ("film_unet", 32),
        ("film_unet", 64),
    ])
    def test_unet_variants(self, model_type, init_features):
        """Test création UNet et variants."""
        kwargs = {
            "in_channels": 1,
            "out_channels": 1,
            "init_features": init_features,
        }

        if "film" in model_type:
            kwargs["condition_size"] = 1

        model = create_model(model_type, **kwargs)
        assert model is not None
        assert hasattr(model, 'forward')

    @pytest.mark.parametrize("base_channels", [32, 64])
    def test_deeplabv3plus(self, base_channels):
        """Test création DeepLabV3+."""
        model = create_model(
            "deeplabv3_plus",
            in_channels=1,
            out_channels=1,
            base_channels=base_channels
        )
        assert model is not None
        assert hasattr(model, 'forward')

    def test_regression_unet(self):
        """Test création UNet regression."""
        model = create_model(
            "regression_unet",
            in_channels=1,
            out_channels=1,
            init_features=64
        )
        assert model is not None
        assert hasattr(model, 'forward')


class TestModelForward:
    """Test le forward pass de tous les modèles."""

    @pytest.fixture
    def input_tensor(self):
        """Tensor d'entrée standard pour tests."""
        return torch.randn(2, 1, 256, 256)

    @pytest.fixture
    def threshold_tensor(self):
        """Tensor threshold pour modèles FiLM."""
        return torch.full((2, 1), 5.0)

    def test_unet_forward(self, input_tensor):
        """Test forward UNet standard."""
        model = create_model("unet", in_channels=1, out_channels=1, init_features=32)
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (2, 1, 256, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_film_unet_forward(self, input_tensor, threshold_tensor):
        """Test forward FiLM-UNet avec threshold."""
        model = create_model(
            "film_unet",
            in_channels=1,
            out_channels=1,
            init_features=32,
            condition_size=1
        )
        model.eval()

        with torch.no_grad():
            output = model(input_tensor, threshold_tensor)

        assert output.shape == (2, 1, 256, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_deeplabv3plus_forward(self, input_tensor):
        """Test forward DeepLabV3+."""
        model = create_model(
            "deeplabv3_plus",
            in_channels=1,
            out_channels=1,
            base_channels=32
        )
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (2, 1, 256, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_regression_unet_forward(self, input_tensor):
        """Test forward UNet regression."""
        model = create_model(
            "regression_unet",
            in_channels=1,
            out_channels=1,
            init_features=32
        )
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (2, 1, 256, 256)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestModelGradients:
    """Test que les gradients se propagent correctement."""

    def test_unet_gradients(self):
        """Test propagation gradients UNet."""
        model = create_model("unet", in_channels=1, out_channels=1, init_features=32)
        model.train()

        input_tensor = torch.randn(2, 1, 256, 256, requires_grad=True)
        target = torch.randint(0, 2, (2, 1, 256, 256)).float()

        output = model(input_tensor)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
        loss.backward()

        # Vérifier que les gradients existent
        assert any(p.grad is not None for p in model.parameters())

        # Vérifier que les gradients ne sont pas tous zéro
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert any(g > 0 for g in grad_norms)

    def test_film_unet_gradients(self):
        """Test propagation gradients FiLM-UNet."""
        model = create_model(
            "film_unet",
            in_channels=1,
            out_channels=1,
            init_features=32,
            condition_size=1
        )
        model.train()

        input_tensor = torch.randn(2, 1, 256, 256, requires_grad=True)
        threshold_tensor = torch.full((2, 1), 5.0)
        target = torch.randint(0, 2, (2, 1, 256, 256)).float()

        output = model(input_tensor, threshold_tensor)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
        loss.backward()

        assert any(p.grad is not None for p in model.parameters())
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert any(g > 0 for g in grad_norms)


class TestModelOutputRange:
    """Test que les outputs sont dans les bonnes plages."""

    @pytest.fixture
    def input_tensor(self):
        return torch.randn(2, 1, 256, 256)

    def test_unet_logits(self, input_tensor):
        """Test que UNet output des logits (pas activés)."""
        model = create_model("unet", in_channels=1, out_channels=1, init_features=32)
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)

        # Logits peuvent être n'importe quelle valeur
        assert output.min() < 0 or output.max() > 1  # Pas déjà sigmoid

    def test_sigmoid_output(self, input_tensor):
        """Test output après sigmoid dans [0, 1]."""
        model = create_model("unet", in_channels=1, out_channels=1, init_features=32)
        model.eval()

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)

        assert probs.min() >= 0
        assert probs.max() <= 1


class TestModelSizes:
    """Test que les modèles gèrent différentes tailles d'images."""

    @pytest.mark.parametrize("size", [128, 256, 512])
    def test_different_input_sizes(self, size):
        """Test différentes tailles d'entrée."""
        model = create_model("unet", in_channels=1, out_channels=1, init_features=32)
        model.eval()

        input_tensor = torch.randn(1, 1, size, size)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (1, 1, size, size)


class TestModelDevice:
    """Test que les modèles fonctionnent sur CPU/GPU."""

    def test_cpu_execution(self):
        """Test exécution sur CPU."""
        model = create_model("unet", in_channels=1, out_channels=1, init_features=32)
        model = model.to("cpu")

        input_tensor = torch.randn(1, 1, 256, 256).to("cpu")

        with torch.no_grad():
            output = model(input_tensor)

        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_execution(self):
        """Test exécution sur GPU."""
        model = create_model("unet", in_channels=1, out_channels=1, init_features=32)
        model = model.to("cuda")

        input_tensor = torch.randn(1, 1, 256, 256).to("cuda")

        with torch.no_grad():
            output = model(input_tensor)

        assert output.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
