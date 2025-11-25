"""
Tests unitaires pour le module de configuration.

Ce module contient des tests pour les fonctionnalités de base du module de configuration,
y compris le chargement, la validation et la sauvegarde des configurations.
"""

import os
import tempfile
from pathlib import Path
import pytest
import yaml

from forestgaps.config.base import forestgaps.configurationManager, ConfigValidationError

# ===================================================================================================
# Tests pour ConfigurationManager
# ===================================================================================================

class TestConfigurationManager:
    """Tests pour la classe ConfigurationManager."""

    def setup_method(self):
        """Préparer les données de test avant chaque test."""
        self.valid_config = {
            "environment": {
                "type": "local",
                "gpu_enabled": True
            },
            "data": {
                "processed_dir": "/tmp/processed",
                "tile_size": 256,
                "overlap": 0.2
            },
            "model": {
                "type": "unet",
                "dropout_rate": 0.2
            },
            "training": {
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001
            }
        }
        
    def test_init(self):
        """Tester l'initialisation avec une configuration valide."""
        config_manager = ConfigurationManager(self.valid_config)
        assert config_manager.config is not None
        assert config_manager.config["environment"]["type"] == "local"
        assert config_manager.config["model"]["type"] == "unet"
        
    def test_init_with_validation(self):
        """Tester l'initialisation avec validation."""
        config_manager = ConfigurationManager(self.valid_config, validate=True)
        assert config_manager.config is not None
        
    def test_save_load_config(self):
        """Tester la sauvegarde et le chargement de la configuration."""
        config_manager = ConfigurationManager(self.valid_config)
        
        # Créer un fichier temporaire pour le test
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp_file:
            temp_filepath = tmp_file.name
            
        try:
            # Sauvegarder la configuration
            config_manager.save_config(temp_filepath)
            assert os.path.exists(temp_filepath)
            
            # Créer un nouveau gestionnaire de configuration et charger le fichier
            new_config_manager = ConfigurationManager()
            new_config_manager.load_config(temp_filepath)
            
            # Vérifier que la configuration chargée correspond à l'originale
            assert new_config_manager.config == self.valid_config
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
                
    def test_get_config_value(self):
        """Tester la récupération de valeurs de configuration."""
        config_manager = ConfigurationManager(self.valid_config)
        
        # Tester la récupération de valeurs existantes
        assert config_manager.get("environment.type") == "local"
        assert config_manager.get("model.dropout_rate") == 0.2
        
        # Tester avec valeur par défaut
        assert config_manager.get("nonexistent.key", "default_value") == "default_value"
        
    def test_set_config_value(self):
        """Tester la modification de valeurs de configuration."""
        config_manager = ConfigurationManager(self.valid_config)
        
        # Modifier une valeur existante
        config_manager.set("model.dropout_rate", 0.5)
        assert config_manager.get("model.dropout_rate") == 0.5
        
        # Ajouter une nouvelle valeur
        config_manager.set("model.new_param", "test_value")
        assert config_manager.get("model.new_param") == "test_value"

# ===================================================================================================
# Tests pour les fonctions d'assistance
# ===================================================================================================

def test_config_validation_error():
    """Tester que l'erreur de validation est levée pour une configuration invalide."""
    invalid_config = {
        "environment": {
            "type": "unknown_type"  # Type d'environnement non valide
        }
    }
    
    with pytest.raises(ConfigValidationError):
        ConfigurationManager(invalid_config, validate=True) 