"""
Tests d'intégration pour le workflow complet.

Vérifie que:
- Le workflow complet fonctionne (préparation → training → évaluation → inférence)
- Les différentes configurations fonctionnent ensemble
- Les modèles peuvent être sauvegardés et rechargés
"""

import pytest
import torch
import tempfile
from pathlib import Path


class TestEndToEndWorkflow:
    """Test le workflow complet end-to-end."""

    def test_create_train_save_load(self):
        """Test création → training → sauvegarde → chargement."""
        from forestgaps.models import create_model

        # 1. Créer modèle
        model = create_model("unet", in_channels=1, out_channels=1, init_features=16)

        # 2. Training step simple
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()

        inputs = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 3. Sauvegarder
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

            # 4. Charger dans nouveau modèle
            new_model = create_model("unet", in_channels=1, out_channels=1, init_features=16)
            checkpoint = torch.load(checkpoint_path)
            new_model.load_state_dict(checkpoint['model_state_dict'])

            # 5. Vérifier que les poids sont identiques
            new_model.eval()
            model.eval()
            with torch.no_grad():
                out1 = model(inputs)
                out2 = new_model(inputs)

            assert torch.allclose(out1, out2, rtol=1e-5)

    def test_film_model_workflow(self):
        """Test workflow complet avec modèle FiLM."""
        from forestgaps.models import create_model

        # Créer modèle FiLM
        model = create_model("film_unet", in_channels=1, out_channels=1,
                           init_features=16, condition_size=1)

        # Training
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()

        inputs = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        threshold = torch.full((2, 1), 5.0)

        optimizer.zero_grad()
        outputs = model(inputs, threshold)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Vérifications
        assert loss.item() >= 0

        # Inférence
        model.eval()
        with torch.no_grad():
            preds = model(inputs, threshold)

        assert preds.shape == targets.shape


class TestConfigWorkflow:
    """Test le workflow avec configurations."""

    def test_load_config_create_model(self):
        """Test chargement config → création modèle."""
        from forestgaps.config import load_training_config
        from forestgaps.models import create_model

        # Charger config
        config = load_training_config("configs/test/minimal.yaml")

        # Extraire config modèle
        model_config = config.model

        # Créer modèle à partir de config
        model_type = model_config.model_type
        if model_type == "unet_film":
            model_type = "film_unet"

        model_kwargs = {
            "in_channels": model_config.in_channels,
            "out_channels": model_config.out_channels,
        }

        if model_type == "unet":
            model_kwargs["init_features"] = model_config.base_channels
        elif model_type == "film_unet":
            model_kwargs["init_features"] = model_config.base_channels
            model_kwargs["condition_size"] = model_config.num_conditions

        model = create_model(model_type, **model_kwargs)

        assert model is not None

    def test_config_compatibility(self):
        """Test compatibilité entre configs."""
        from forestgaps.config import load_training_config

        configs = [
            "configs/test/minimal.yaml",
            "configs/test/quick.yaml",
        ]

        loaded_configs = []
        for config_path in configs:
            try:
                config = load_training_config(config_path)
                loaded_configs.append(config)
            except FileNotFoundError:
                pytest.skip(f"Config {config_path} not found")

        # Toutes les configs doivent avoir les mêmes champs essentiels
        for config in loaded_configs:
            assert hasattr(config, 'model')
            assert hasattr(config, 'epochs')
            assert hasattr(config, 'batch_size')


class TestMultiModelWorkflow:
    """Test workflow avec plusieurs modèles."""

    @pytest.mark.parametrize("model_type", [
        "unet",
        "film_unet",
        "deeplabv3_plus",
    ])
    def test_model_training_workflow(self, model_type):
        """Test workflow training pour différents modèles."""
        from forestgaps.models import create_model

        # Créer modèle
        kwargs = {"in_channels": 1, "out_channels": 1}
        if model_type == "unet":
            kwargs["init_features"] = 16
        elif model_type == "film_unet":
            kwargs["init_features"] = 16
            kwargs["condition_size"] = 1
        elif model_type == "deeplabv3_plus":
            kwargs["encoder_name"] = "resnet18"
            kwargs["encoder_weights"] = None

        model = create_model(model_type, **kwargs)

        # Setup training
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()

        inputs = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

        # Training step
        optimizer.zero_grad()

        # Forward (avec threshold pour FiLM)
        if 'film' in model_type:
            threshold = torch.full((2, 1), 5.0)
            outputs = model(inputs, threshold)
        else:
            outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Vérifications
        assert loss.item() >= 0
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestDataWorkflow:
    """Test workflow de préparation des données."""

    def test_data_loading_concept(self):
        """Test concept de chargement de données."""
        from torch.utils.data import Dataset, DataLoader

        class DummyDataset(Dataset):
            def __init__(self, n_samples=10):
                self.n_samples = n_samples

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                return {
                    'image': torch.randn(1, 64, 64),
                    'mask': torch.randint(0, 2, (1, 64, 64)).float()
                }

        # Créer dataset et loader
        dataset = DummyDataset(n_samples=20)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Itérer
        for batch in loader:
            assert batch['image'].shape == (4, 1, 64, 64)
            assert batch['mask'].shape == (4, 1, 64, 64)
            break  # Juste un batch pour test


class TestOptimizationWorkflow:
    """Test workflow d'optimisation."""

    def test_amp_workflow(self):
        """Test workflow avec AMP."""
        from forestgaps.models import create_model

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device('cuda')
        model = create_model("unet", in_channels=1, out_channels=1, init_features=16)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler()

        inputs = torch.randn(2, 1, 64, 64).to(device)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float().to(device)

        # Training step avec AMP
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        assert loss.item() >= 0

    def test_scheduler_workflow(self):
        """Test workflow avec scheduler."""
        from forestgaps.models import create_model

        model = create_model("unet", in_channels=1, out_channels=1, init_features=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        initial_lr = optimizer.param_groups[0]['lr']

        # Plusieurs steps
        for epoch in range(10):
            # Training step dummy
            inputs = torch.randn(2, 1, 64, 64)
            targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
            loss.backward()
            optimizer.step()

            # Step scheduler
            scheduler.step()

        # LR devrait avoir changé
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr < initial_lr


class TestErrorHandling:
    """Test gestion des erreurs."""

    def test_invalid_model_type(self):
        """Test création modèle avec type invalide."""
        from forestgaps.models import create_model

        with pytest.raises((ValueError, KeyError)):
            create_model("invalid_model_type")

    def test_mismatched_dimensions(self):
        """Test erreur dimensions incompatibles."""
        from forestgaps.models import create_model

        model = create_model("unet", in_channels=1, out_channels=1, init_features=16)

        # Input avec mauvais nombre de canaux
        wrong_inputs = torch.randn(2, 3, 64, 64)  # 3 canaux au lieu de 1

        with pytest.raises(RuntimeError):
            model(wrong_inputs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
