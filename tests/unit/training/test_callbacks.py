"""
Tests unitaires pour le système de callbacks d'entraînement.

Ce module contient des tests pour les différents callbacks utilisés 
pendant l'entraînement des modèles.
"""

import os
import tempfile
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, call

from forestgaps.training.callbacks import (
    Callback,
    ModelCheckpointCallback,
    EarlyStoppingCallback,
    LearningRateSchedulerCallback,
    TensorBoardCallback,
    CallbackList
)

# ===================================================================================================
# Tests pour la classe Callback de base
# ===================================================================================================

class TestCallback:
    """Tests pour la classe Callback de base."""
    
    def test_base_implementation(self):
        """Tester que les méthodes de base sont implémentées."""
        callback = Callback()
        
        # Vérifier que l'appel des méthodes ne génère pas d'erreur
        callback.on_train_begin({})
        callback.on_train_end({})
        callback.on_epoch_begin(1, {})
        callback.on_epoch_end(1, {})
        callback.on_batch_begin(1, {})
        callback.on_batch_end(1, {})
        callback.on_validation_begin({})
        callback.on_validation_end({})
        
        # La classe de base ne fait rien, donc ces appels ne modifient pas l'état

# ===================================================================================================
# Tests pour ModelCheckpointCallback
# ===================================================================================================

class TestModelCheckpointCallback:
    """Tests pour ModelCheckpointCallback."""
    
    def setup_method(self):
        """Préparer l'environnement pour les tests."""
        self.model = MagicMock()
        self.optimizer = MagicMock()
        
        # Créer un répertoire temporaire pour sauvegarder les modèles
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_dir = self.temp_dir.name
        
    def teardown_method(self):
        """Nettoyer après les tests."""
        self.temp_dir.cleanup()
    
    @patch('torch.save')
    def test_save_best_only(self, mock_save):
        """Tester la sauvegarde du meilleur modèle uniquement."""
        # Configurer le callback
        callback = ModelCheckpointCallback(
            save_dir=self.save_dir,
            save_best_only=True,
            metric_name="val_iou",
            mode="max"
        )
        
        # Simuler la fin d'une époque avec une métrique améliorée
        logs = {"val_iou": 0.8}
        callback.on_train_begin({})
        callback.on_epoch_end(1, logs, self.model, self.optimizer)
        
        # Vérifier que le modèle a été sauvegardé
        assert mock_save.called
        
        # Réinitialiser le mock
        mock_save.reset_mock()
        
        # Simuler la fin d'une époque avec une métrique inférieure
        logs = {"val_iou": 0.7}
        callback.on_epoch_end(2, logs, self.model, self.optimizer)
        
        # Vérifier que le modèle n'a pas été sauvegardé
        assert not mock_save.called
        
        # Simuler la fin d'une époque avec une métrique améliorée
        logs = {"val_iou": 0.9}
        callback.on_epoch_end(3, logs, self.model, self.optimizer)
        
        # Vérifier que le modèle a été sauvegardé
        assert mock_save.called
    
    @patch('torch.save')
    def test_save_all_epochs(self, mock_save):
        """Tester la sauvegarde du modèle à chaque époque."""
        # Configurer le callback pour sauvegarder à chaque époque
        callback = ModelCheckpointCallback(
            save_dir=self.save_dir,
            save_best_only=False,
            metric_name="val_iou"
        )
        
        # Simuler la fin de plusieurs époques
        logs = {"val_iou": 0.7}
        callback.on_train_begin({})
        
        for epoch in range(1, 4):
            callback.on_epoch_end(epoch, logs, self.model, self.optimizer)
            
            # Vérifier que le modèle a été sauvegardé à chaque époque
            assert mock_save.call_count == epoch

# ===================================================================================================
# Tests pour EarlyStoppingCallback
# ===================================================================================================

class TestEarlyStoppingCallback:
    """Tests pour EarlyStoppingCallback."""
    
    def test_early_stopping_improvement(self):
        """Tester que l'arrêt précoce ne se déclenche pas lors d'améliorations."""
        # Configurer le callback avec une patience de 2 époques
        callback = EarlyStoppingCallback(patience=2, metric_name="val_loss", mode="min")
        callback.on_train_begin({})
        
        # Simuler plusieurs époques avec des améliorations continues
        logs = {"val_loss": 1.0}
        should_stop = callback.on_epoch_end(1, logs)
        assert not should_stop
        
        logs = {"val_loss": 0.9}
        should_stop = callback.on_epoch_end(2, logs)
        assert not should_stop
        
        logs = {"val_loss": 0.8}
        should_stop = callback.on_epoch_end(3, logs)
        assert not should_stop
    
    def test_early_stopping_trigger(self):
        """Tester que l'arrêt précoce se déclenche après la patience épuisée."""
        # Configurer le callback avec une patience de 2 époques
        callback = EarlyStoppingCallback(patience=2, metric_name="val_loss", mode="min")
        callback.on_train_begin({})
        
        # Simuler l'époque 1 avec une métrique initiale
        logs = {"val_loss": 1.0}
        should_stop = callback.on_epoch_end(1, logs)
        assert not should_stop
        
        # Simuler l'époque 2 avec une métrique qui ne s'améliore pas
        logs = {"val_loss": 1.1}
        should_stop = callback.on_epoch_end(2, logs)
        assert not should_stop
        
        # Simuler l'époque 3 avec une métrique qui ne s'améliore toujours pas
        logs = {"val_loss": 1.2}
        should_stop = callback.on_epoch_end(3, logs)
        assert not should_stop
        
        # Simuler l'époque 4 avec une métrique qui ne s'améliore toujours pas
        # La patience devrait être épuisée (2 époques sans amélioration)
        logs = {"val_loss": 1.3}
        should_stop = callback.on_epoch_end(4, logs)
        assert should_stop

# ===================================================================================================
# Tests pour LearningRateSchedulerCallback
# ===================================================================================================

class TestLearningRateSchedulerCallback:
    """Tests pour LearningRateSchedulerCallback."""
    
    def setup_method(self):
        """Préparer l'environnement pour les tests."""
        self.optimizer = MagicMock()
        self.optimizer.param_groups = [{"lr": 0.001}]
    
    def test_step_lr_scheduler(self):
        """Tester le scheduler StepLR."""
        # Configurer le callback
        callback = LearningRateSchedulerCallback(
            mode="step",
            step_size=2,
            gamma=0.1
        )
        callback.on_train_begin({"optimizer": self.optimizer})
        
        # Vérifier le LR initial
        assert self.optimizer.param_groups[0]["lr"] == 0.001
        
        # Simuler la fin des époques
        for epoch in range(1, 6):
            callback.on_epoch_end(epoch, {}, None, self.optimizer)
            
            # Vérifier que le LR est ajusté tous les 2 époques
            expected_lr = 0.001 * (0.1 ** (epoch // 2))
            assert abs(self.optimizer.param_groups[0]["lr"] - expected_lr) < 1e-6
    
    def test_cosine_annealing_scheduler(self):
        """Tester le scheduler CosineAnnealingLR."""
        # Configurer le callback
        callback = LearningRateSchedulerCallback(
            mode="cosine",
            T_max=5
        )
        callback.on_train_begin({"optimizer": self.optimizer})
        
        # Vérifier le LR initial
        assert self.optimizer.param_groups[0]["lr"] == 0.001
        
        # Simuler la fin des époques (les valeurs exactes de LR dépendent de la formule cosinus)
        for epoch in range(1, 6):
            callback.on_epoch_end(epoch, {}, None, self.optimizer)
            
            # Vérifier simplement que le LR est modifié
            assert self.optimizer.param_groups[0]["lr"] <= 0.001

# ===================================================================================================
# Tests pour TensorBoardCallback
# ===================================================================================================

class TestTensorBoardCallback:
    """Tests pour TensorBoardCallback."""
    
    def setup_method(self):
        """Préparer l'environnement pour les tests."""
        # Créer un répertoire temporaire pour les logs TensorBoard
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = self.temp_dir.name
    
    def teardown_method(self):
        """Nettoyer après les tests."""
        self.temp_dir.cleanup()
    
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_initialization(self, mock_writer):
        """Tester l'initialisation du callback TensorBoard."""
        callback = TensorBoardCallback(log_dir=self.log_dir)
        callback.on_train_begin({})
        
        # Vérifier que le SummaryWriter a été créé
        mock_writer.assert_called_once_with(log_dir=self.log_dir)
    
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_log_metrics(self, mock_writer):
        """Tester la journalisation des métriques dans TensorBoard."""
        # Configurer le mock
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        
        # Créer le callback et initialiser
        callback = TensorBoardCallback(log_dir=self.log_dir)
        callback.on_train_begin({})
        
        # Simuler la fin d'une époque avec des métriques
        logs = {
            "loss": 0.5,
            "val_loss": 0.6,
            "val_iou": 0.8
        }
        callback.on_epoch_end(1, logs)
        
        # Vérifier que add_scalar a été appelé pour chaque métrique
        mock_writer_instance.add_scalar.assert_any_call("loss", 0.5, 1)
        mock_writer_instance.add_scalar.assert_any_call("val_loss", 0.6, 1)
        mock_writer_instance.add_scalar.assert_any_call("val_iou", 0.8, 1)
    
    @patch('torch.utils.tensorboard.SummaryWriter')
    def test_cleanup(self, mock_writer):
        """Tester le nettoyage à la fin de l'entraînement."""
        # Configurer le mock
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        
        # Créer le callback et initialiser
        callback = TensorBoardCallback(log_dir=self.log_dir)
        callback.on_train_begin({})
        
        # Simuler la fin de l'entraînement
        callback.on_train_end({})
        
        # Vérifier que close a été appelé
        mock_writer_instance.close.assert_called_once()

# ===================================================================================================
# Tests pour CallbackList
# ===================================================================================================

class TestCallbackList:
    """Tests pour la classe CallbackList."""
    
    def test_callback_aggregation(self):
        """Tester l'agrégation des callbacks."""
        # Créer des mocks pour les callbacks
        callback1 = MagicMock(spec=Callback)
        callback2 = MagicMock(spec=Callback)
        
        # Créer la liste de callbacks
        callback_list = CallbackList([callback1, callback2])
        
        # Simuler le début de l'entraînement
        logs = {}
        callback_list.on_train_begin(logs)
        
        # Vérifier que chaque callback a été appelé
        callback1.on_train_begin.assert_called_once_with(logs)
        callback2.on_train_begin.assert_called_once_with(logs)
    
    def test_early_stopping_propagation(self):
        """Tester la propagation de l'arrêt précoce."""
        # Créer un mock pour un callback régulier
        regular_callback = MagicMock(spec=Callback)
        regular_callback.on_epoch_end.return_value = False
        
        # Créer un mock pour un callback d'arrêt précoce
        early_stopping_callback = MagicMock(spec=EarlyStoppingCallback)
        early_stopping_callback.on_epoch_end.return_value = True  # Signaler l'arrêt
        
        # Créer la liste de callbacks
        callback_list = CallbackList([regular_callback, early_stopping_callback])
        
        # Simuler la fin d'une époque
        logs = {}
        result = callback_list.on_epoch_end(1, logs, None, None)
        
        # Vérifier que l'arrêt précoce est propagé
        assert result is True
        
    def test_callback_execution_order(self):
        """Tester l'ordre d'exécution des callbacks."""
        # Créer un objet pour tracer l'ordre d'exécution
        execution_order = []
        
        class TraceCallback(Callback):
            def __init__(self, name):
                self.name = name
                
            def on_epoch_begin(self, epoch, logs=None):
                execution_order.append(self.name)
        
        # Créer les callbacks
        callback1 = TraceCallback("callback1")
        callback2 = TraceCallback("callback2")
        callback3 = TraceCallback("callback3")
        
        # Créer la liste de callbacks
        callback_list = CallbackList([callback1, callback2, callback3])
        
        # Simuler le début d'une époque
        callback_list.on_epoch_begin(1, {})
        
        # Vérifier l'ordre d'exécution
        assert execution_order == ["callback1", "callback2", "callback3"] 