"""
Tests unitaires pour le système de configuration.

Vérifie que:
- Les configs se chargent correctement
- La validation Pydantic fonctionne
- Les configs par défaut sont valides
"""

import pytest
from pathlib import Path
from forestgaps.config import (
    load_training_config,
    load_data_config,
    load_model_config,
    load_default_config,
)


class TestConfigLoading:
    """Test le chargement des configurations."""

    def test_load_minimal_config(self):
        """Test chargement config minimal."""
        config = load_training_config("configs/test/minimal.yaml")
        assert config is not None
        assert config.epochs == 2
        assert config.batch_size == 4

    def test_load_quick_config(self):
        """Test chargement config quick."""
        config = load_training_config("configs/test/quick.yaml")
        assert config is not None
        assert config.epochs == 5
        assert config.batch_size == 8

    def test_load_data_config(self):
        """Test chargement config data."""
        config = load_data_config("configs/test/data_quick.yaml")
        assert config is not None
        assert hasattr(config, 'preprocessing')
        assert config.preprocessing.tile_size > 0

    def test_load_model_config(self):
        """Test chargement config model."""
        config = load_model_config("configs/test/model_quick.yaml")
        assert config is not None
        assert config.model_type in ["unet", "film_unet", "deeplabv3_plus"]
        assert config.in_channels > 0
        assert config.out_channels > 0

    def test_load_default_config(self):
        """Test chargement config par défaut."""
        config = load_default_config()
        assert config is not None
        # Config par défaut doit avoir tous les champs nécessaires


class TestConfigValidation:
    """Test la validation Pydantic des configs."""

    def test_training_config_validation(self):
        """Test validation config training."""
        config = load_training_config("configs/test/minimal.yaml")

        # Epochs doit être positif
        assert config.epochs > 0

        # Batch size doit être positif
        assert config.batch_size > 0

        # Learning rate doit être positive
        assert config.optimizer.lr > 0

    def test_model_config_validation(self):
        """Test validation config model."""
        config = load_model_config("configs/test/model_quick.yaml")

        # Channels doivent être positifs
        assert config.in_channels > 0
        assert config.out_channels > 0
        assert config.base_channels > 0

        # Depth doit être raisonnable
        assert 1 <= config.depth <= 5

    def test_data_config_validation(self):
        """Test validation config data."""
        config = load_data_config("configs/test/data_quick.yaml")

        # Tile size doit être positif et multiple de 2
        assert config.preprocessing.tile_size > 0
        assert config.preprocessing.tile_size % 2 == 0


class TestConfigDefaults:
    """Test que les valeurs par défaut sont sensées."""

    def test_optimizer_defaults(self):
        """Test valeurs par défaut optimizer."""
        config = load_training_config("configs/test/quick.yaml")

        assert config.optimizer.type in ["adam", "adamw", "sgd"]
        assert 0 < config.optimizer.lr < 1
        assert 0 <= config.optimizer.weight_decay < 1

    def test_scheduler_defaults(self):
        """Test valeurs par défaut scheduler."""
        config = load_training_config("configs/test/quick.yaml")

        assert config.scheduler.type in ["onecycle", "reduce_on_plateau", "step", "cosine"]

    def test_loss_defaults(self):
        """Test valeurs par défaut loss."""
        config = load_training_config("configs/test/quick.yaml")

        if config.loss.type == "combo":
            # Poids doivent être non-négatifs
            assert config.loss.bce_weight >= 0
            assert config.loss.dice_weight >= 0
            assert config.loss.focal_weight >= 0

            # Au moins un poids non-zéro
            assert (config.loss.bce_weight +
                   config.loss.dice_weight +
                   config.loss.focal_weight) > 0


class TestConfigCompatibility:
    """Test compatibilité entre configs."""

    def test_model_type_compatibility(self):
        """Test que model_type est compatible avec registry."""
        config = load_model_config("configs/test/model_quick.yaml")

        # Model types supportés
        valid_types = [
            "unet", "film_unet", "unet_film",
            "deeplabv3_plus", "res_unet", "res_unet_film",
            "regression_unet"
        ]

        assert config.model_type in valid_types

    def test_film_model_has_conditions(self):
        """Test que modèles FiLM ont num_conditions."""
        config = load_model_config("configs/test/model_quick.yaml")

        if "film" in config.model_type:
            assert hasattr(config, 'num_conditions')
            assert config.num_conditions > 0


class TestConfigPaths:
    """Test que les chemins dans les configs existent."""

    def test_config_files_exist(self):
        """Test que les fichiers de config existent."""
        config_files = [
            "configs/test/minimal.yaml",
            "configs/test/quick.yaml",
            "configs/test/data_minimal.yaml",
            "configs/test/data_quick.yaml",
            "configs/test/model_minimal.yaml",
            "configs/test/model_quick.yaml",
        ]

        for config_file in config_files:
            assert Path(config_file).exists(), f"Config file not found: {config_file}"


class TestConfigRoundtrip:
    """Test que les configs peuvent être sauvegardées et rechargées."""

    def test_training_config_roundtrip(self, tmp_path):
        """Test save/load config training."""
        # Charger config
        config = load_training_config("configs/test/quick.yaml")

        # Sauvegarder
        output_path = tmp_path / "test_config.yaml"

        # Utiliser model_dump au lieu de dict si Pydantic V2
        if hasattr(config, 'model_dump'):
            config_dict = config.model_dump()
        else:
            config_dict = config.dict()

        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f)

        # Recharger
        reloaded_config = load_training_config(str(output_path))

        # Comparer quelques valeurs clés
        assert reloaded_config.epochs == config.epochs
        assert reloaded_config.batch_size == config.batch_size
        assert reloaded_config.optimizer.lr == config.optimizer.lr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
