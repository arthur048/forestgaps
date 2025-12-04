"""
Tests unitaires pour le module training.

Vérifie que:
- Les loss functions fonctionnent
- Les schedulers fonctionnent
- Le training loop de base fonctionne
"""

import pytest
import torch
from forestgaps.training.losses import ComboLoss, DiceLoss, FocalLoss


class TestComboLoss:
    """Test la Combo Loss (BCE + Dice + Focal)."""

    @pytest.fixture
    def sample_data(self):
        """Données de test pour loss."""
        logits = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        return logits, targets

    def test_combo_loss_forward(self, sample_data):
        """Test forward pass Combo Loss."""
        logits, targets = sample_data

        loss_fn = ComboLoss(bce_weight=1.0, dice_weight=1.0, focal_weight=1.0)
        loss, breakdown = loss_fn(logits, targets)

        # Loss doit être un scalaire
        assert loss.ndim == 0

        # Loss doit être positive
        assert loss.item() >= 0

        # Breakdown doit contenir les 3 composants
        assert 'bce_loss' in breakdown
        assert 'dice_loss' in breakdown
        assert 'focal_loss' in breakdown

    def test_combo_loss_weights(self, sample_data):
        """Test que les poids de Combo Loss fonctionnent."""
        logits, targets = sample_data

        # Loss avec seulement BCE
        loss_bce_only = ComboLoss(bce_weight=1.0, dice_weight=0.0, focal_weight=0.0)
        loss_bce, breakdown_bce = loss_bce_only(logits, targets)

        # Loss avec seulement Dice
        loss_dice_only = ComboLoss(bce_weight=0.0, dice_weight=1.0, focal_weight=0.0)
        loss_dice, breakdown_dice = loss_dice_only(logits, targets)

        # Les pertes doivent être différentes
        assert not torch.isclose(loss_bce, loss_dice, rtol=0.1)

    def test_combo_loss_gradients(self, sample_data):
        """Test que Combo Loss produit des gradients."""
        logits, targets = sample_data
        logits.requires_grad = True

        loss_fn = ComboLoss(bce_weight=1.0, dice_weight=1.0, focal_weight=1.0)
        loss, _ = loss_fn(logits, targets)

        loss.backward()

        # Gradients doivent exister
        assert logits.grad is not None

        # Gradients ne doivent pas être tous zéro
        assert logits.grad.abs().sum() > 0

    def test_combo_loss_zero_weights(self, sample_data):
        """Test Combo Loss avec poids zéro."""
        logits, targets = sample_data

        loss_fn = ComboLoss(bce_weight=0.0, dice_weight=0.0, focal_weight=0.0)
        loss, _ = loss_fn(logits, targets)

        # Avec poids zéro, loss devrait être zéro
        assert loss.item() == 0.0


class TestDiceLoss:
    """Test la Dice Loss."""

    def test_dice_loss_perfect_prediction(self):
        """Test Dice Loss avec prédiction parfaite."""
        # Prédiction parfaite
        logits = torch.ones(2, 1, 64, 64) * 10  # Sigmoid -> ~1
        targets = torch.ones(2, 1, 64, 64)

        loss_fn = DiceLoss()
        loss = loss_fn(logits, targets)

        # Loss devrait être proche de 0
        assert loss.item() < 0.1

    def test_dice_loss_worst_prediction(self):
        """Test Dice Loss avec pire prédiction."""
        # Pire prédiction
        logits = torch.ones(2, 1, 64, 64) * 10  # Sigmoid -> ~1
        targets = torch.zeros(2, 1, 64, 64)      # Mais target est 0

        loss_fn = DiceLoss()
        loss = loss_fn(logits, targets)

        # Loss devrait être proche de 1
        assert loss.item() > 0.8


class TestFocalLoss:
    """Test la Focal Loss."""

    def test_focal_loss_forward(self):
        """Test forward Focal Loss."""
        logits = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        loss = loss_fn(logits, targets)

        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_focal_loss_gamma_effect(self):
        """Test que gamma affecte la loss."""
        logits = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

        # Focal Loss avec gamma faible
        loss_fn_low = FocalLoss(alpha=0.25, gamma=1.0)
        loss_low = loss_fn_low(logits, targets)

        # Focal Loss avec gamma élevé
        loss_fn_high = FocalLoss(alpha=0.25, gamma=5.0)
        loss_high = loss_fn_high(logits, targets)

        # Les pertes doivent être différentes
        assert not torch.isclose(loss_low, loss_high, rtol=0.1)


class TestTrainingLoop:
    """Test le training loop basique."""

    def test_simple_training_step(self):
        """Test une étape de training simple."""
        from forestgaps.models import create_model

        # Setup
        model = create_model("unet", in_channels=1, out_channels=1, init_features=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Données dummy
        inputs = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

        # Training step
        model.train()
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # Vérifications
        assert loss.item() >= 0
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_training_reduces_loss(self):
        """Test que le training réduit la loss (overfitting test)."""
        from forestgaps.models import create_model

        # Setup
        model = create_model("unet", in_channels=1, out_channels=1, init_features=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Données dummy fixes
        inputs = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

        # Loss initiale
        model.train()
        with torch.no_grad():
            outputs = model(inputs)
            initial_loss = criterion(outputs, targets).item()

        # Quelques steps de training
        for _ in range(10):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Loss finale
        with torch.no_grad():
            outputs = model(inputs)
            final_loss = criterion(outputs, targets).item()

        # Loss devrait diminuer
        assert final_loss < initial_loss


class TestOptimization:
    """Test les composants d'optimisation."""

    def test_gradient_clipping(self):
        """Test gradient clipping."""
        from forestgaps.models import create_model

        model = create_model("unet", in_channels=1, out_channels=1, init_features=16)

        # Générer des gradients
        inputs = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

        outputs = model(inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
        loss.backward()

        # Clipper les gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Vérifier que les gradients existent toujours
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

        # Vérifier que la norme est <= 1.0
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        assert total_norm <= 1.0 or torch.isclose(total_norm, torch.tensor(1.0), rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
