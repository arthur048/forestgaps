"""
Tests d'intégration pour le workflow d'entraînement.

Ce module contient des tests qui valident l'intégration du workflow
d'entraînement complet, de la configuration jusqu'à l'évaluation.
"""

import os
import tempfile
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, ANY

from forestgaps.config import ConfigurationManager
from forestgaps.data.datasets import GapDataset
from forestgaps.data.loaders import create_data_loaders
from forestgaps.models import create_model, ModelRegistry
from forestgaps.models.unet import UNet
from forestgaps.training import Trainer
from forestgaps.training.callbacks import (
    ModelCheckpointCallback,
    EarlyStoppingCallback,
    LearningRateSchedulerCallback
)
from forestgaps.training.metrics import SegmentationMetrics
from forestgaps.environment import setup_environment, Environment

# ===================================================================================================
# Fixtures pour les tests d'intégration du workflow d'entraînement
# ===================================================================================================

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
            "epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "batch_size": 4,
            "gradient_clipping": 1.0,
            "early_stopping_patience": 5,
            "early_stopping_metric": "val_loss",
            "early_stopping_mode": "min"
        },
        "evaluation": {
            "batch_size": 8,
            "metrics": ["iou", "f1", "precision", "recall"]
        }
    }
    return ConfigurationManager(config_data)

@pytest.fixture
def mock_environment():
    """Créer un environnement simulé pour les tests."""
    env = MagicMock(spec=Environment)
    env.name = "test"
    env.has_gpu = False
    env.get_device.return_value = torch.device("cpu")
    env.setup_resources.return_value = None
    return env

@pytest.fixture
def mock_dataloader():
    """Créer un DataLoader simulé pour les tests."""
    # Créer des données simulées
    batch_size = 4
    num_batches = 3
    
    # Simuler plusieurs lots de données
    batches = []
    for _ in range(num_batches):
        dsm = torch.randn(batch_size, 1, 64, 64)
        masks = [torch.randint(0, 2, (batch_size, 1, 64, 64)).float() for _ in range(3)]  # 3 seuils
        batches.append((dsm, masks))
    
    # Créer un DataLoader mock qui retourne ces lots
    dataloader = MagicMock()
    dataloader.__iter__.return_value = iter(batches)
    dataloader.__len__.return_value = num_batches
    
    return dataloader

# ===================================================================================================
# Tests d'intégration du workflow d'entraînement
# ===================================================================================================

class TestTrainingWorkflow:
    """Tests d'intégration pour le workflow d'entraînement complet."""
    
    def setup_method(self):
        """Préparer l'environnement pour les tests."""
        # Créer un répertoire temporaire pour les sorties
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Enregistrer le modèle UNet dans le registre
        ModelRegistry.register("unet", UNet)
    
    def teardown_method(self):
        """Nettoyer après les tests."""
        self.temp_dir.cleanup()
    
    @patch('torch.optim.AdamW')
    @patch('torch.nn.BCEWithLogitsLoss')
    def test_trainer_initialization(self, mock_loss, mock_optimizer, mock_config, mock_environment, mock_dataloader):
        """Tester l'initialisation du Trainer avec tous les composants."""
        # Créer un modèle
        model = create_model(
            model_type="unet",
            **mock_config.get("model.params", {})
        )
        
        # Configurer les callbacks
        callbacks = [
            ModelCheckpointCallback(
                save_dir=self.output_dir,
                save_best_only=True,
                metric_name="val_loss"
            ),
            EarlyStoppingCallback(
                patience=mock_config.get("training.early_stopping_patience"),
                metric_name=mock_config.get("training.early_stopping_metric"),
                mode=mock_config.get("training.early_stopping_mode")
            ),
            LearningRateSchedulerCallback(
                mode="step",
                step_size=1,
                gamma=0.1
            )
        ]
        
        # Créer le Trainer
        trainer = Trainer(
            model=model,
            config=mock_config,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            callbacks=callbacks,
            device=mock_environment.get_device()
        )
        
        # Vérifier que le Trainer a été correctement initialisé
        assert trainer.model is model
        assert trainer.train_loader is mock_dataloader
        assert trainer.val_loader is mock_dataloader
        assert trainer.device == mock_environment.get_device()
        assert len(trainer.callbacks) == 3
        assert mock_optimizer.called
        assert mock_loss.called
    
    @patch('forestgaps.training.Trainer._train_epoch')
    @patch('forestgaps.training.Trainer._validate')
    def test_training_loop(self, mock_validate, mock_train_epoch, mock_config, mock_environment, mock_dataloader):
        """Tester la boucle d'entraînement complète."""
        # Configurer les mocks
        mock_train_epoch.return_value = {"loss": 0.5}
        mock_validate.return_value = {"val_loss": 0.6, "val_iou": 0.7}
        
        # Créer un modèle
        model = create_model(
            model_type="unet",
            **mock_config.get("model.params", {})
        )
        
        # Créer le Trainer sans callbacks pour simplifier
        trainer = Trainer(
            model=model,
            config=mock_config,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            device=mock_environment.get_device()
        )
        
        # Exécuter l'entraînement
        epochs = 2
        results = trainer.train(epochs=epochs)
        
        # Vérifier que la boucle d'entraînement s'est exécutée correctement
        assert mock_train_epoch.call_count == epochs
        assert mock_validate.call_count == epochs
        
        # Vérifier les résultats
        assert "history" in results
        assert len(results["history"]["loss"]) == epochs
        assert len(results["history"]["val_loss"]) == epochs
        assert len(results["history"]["val_iou"]) == epochs
    
    @patch('torch.save')
    def test_model_checkpoint(self, mock_save, mock_config, mock_environment, mock_dataloader):
        """Tester l'intégration du ModelCheckpointCallback dans le workflow d'entraînement."""
        # Créer un modèle
        model = create_model(
            model_type="unet",
            **mock_config.get("model.params", {})
        )
        
        # Configurer le callback de checkpoint
        checkpoint_callback = ModelCheckpointCallback(
            save_dir=self.output_dir,
            save_best_only=True,
            metric_name="val_loss",
            mode="min"
        )
        
        # Créer le Trainer avec le callback
        trainer = Trainer(
            model=model,
            config=mock_config,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            callbacks=[checkpoint_callback],
            device=mock_environment.get_device()
        )
        
        # Patcher les méthodes d'entraînement et de validation pour simuler l'entraînement
        trainer._train_epoch = MagicMock(return_value={"loss": 0.5})
        trainer._validate = MagicMock(side_effect=[
            {"val_loss": 0.6, "val_iou": 0.7},  # Première époque
            {"val_loss": 0.4, "val_iou": 0.8}   # Deuxième époque (meilleure)
        ])
        
        # Exécuter l'entraînement
        trainer.train(epochs=2)
        
        # Vérifier que le modèle a été sauvegardé deux fois
        # Une fois après l'initialisation, une fois après la deuxième époque (meilleure)
        assert mock_save.call_count == 2
    
    def test_early_stopping(self, mock_config, mock_environment, mock_dataloader):
        """Tester l'intégration de l'EarlyStoppingCallback dans le workflow d'entraînement."""
        # Créer un modèle
        model = create_model(
            model_type="unet",
            **mock_config.get("model.params", {})
        )
        
        # Configurer le callback d'arrêt précoce avec une patience de 1 époque
        early_stopping_callback = EarlyStoppingCallback(
            patience=1,
            metric_name="val_loss",
            mode="min"
        )
        
        # Créer le Trainer avec le callback
        trainer = Trainer(
            model=model,
            config=mock_config,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            callbacks=[early_stopping_callback],
            device=mock_environment.get_device()
        )
        
        # Patcher les méthodes d'entraînement et de validation pour simuler l'entraînement
        # Simuler des métriques de validation qui se dégradent
        trainer._train_epoch = MagicMock(return_value={"loss": 0.5})
        trainer._validate = MagicMock(side_effect=[
            {"val_loss": 0.6, "val_iou": 0.7},  # Première époque
            {"val_loss": 0.8, "val_iou": 0.6},  # Deuxième époque (pire)
            {"val_loss": 1.0, "val_iou": 0.5}   # Troisième époque (encore pire)
        ])
        
        # Exécuter l'entraînement avec un maximum de 5 époques
        results = trainer.train(epochs=5)
        
        # Vérifier que l'entraînement s'est arrêté après 3 époques (2 + patience de 1)
        assert len(results["history"]["loss"]) == 3
        assert trainer._train_epoch.call_count == 3
        assert trainer._validate.call_count == 3
    
    def test_learning_rate_scheduler(self, mock_config, mock_environment, mock_dataloader):
        """Tester l'intégration du LearningRateSchedulerCallback dans le workflow d'entraînement."""
        # Créer un modèle
        model = create_model(
            model_type="unet",
            **mock_config.get("model.params", {})
        )
        
        # Configurer le callback de scheduler de taux d'apprentissage
        lr_scheduler_callback = LearningRateSchedulerCallback(
            mode="step",
            step_size=1,
            gamma=0.1
        )
        
        # Créer le Trainer avec le callback
        trainer = Trainer(
            model=model,
            config=mock_config,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            callbacks=[lr_scheduler_callback],
            device=mock_environment.get_device()
        )
        
        # Patcher l'optimiseur pour vérifier les changements de LR
        initial_lr = mock_config.get("training.learning_rate")
        trainer.optimizer = MagicMock()
        trainer.optimizer.param_groups = [{"lr": initial_lr}]
        
        # Patcher les méthodes d'entraînement et de validation pour simuler l'entraînement
        trainer._train_epoch = MagicMock(return_value={"loss": 0.5})
        trainer._validate = MagicMock(return_value={"val_loss": 0.6, "val_iou": 0.7})
        
        # Exécuter l'entraînement
        trainer.train(epochs=3)
        
        # Vérifier que le LR a été ajusté correctement
        # Époque 1: initial_lr
        # Époque 2: initial_lr * 0.1
        # Époque 3: initial_lr * 0.1 * 0.1
        expected_lrs = [initial_lr, initial_lr * 0.1, initial_lr * 0.01]
        
        # Vérifier que le LR a été modifié le bon nombre de fois
        assert trainer.optimizer.param_groups[0]["lr"] == expected_lrs[-1] 