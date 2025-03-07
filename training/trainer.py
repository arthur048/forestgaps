"""
Module de la classe Trainer pour l'entraînement des modèles.

Ce module fournit la classe principale pour l'entraînement des modèles
de segmentation pour la détection des trouées forestières.
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from forestgaps_dl.config import Config
from forestgaps_dl.environment import get_device

from .metrics.segmentation import SegmentationMetrics
from .metrics.classification import ThresholdMetrics
from .loss.combined import CombinedFocalDiceLoss
from .loss.factory import create_loss_function, create_loss_with_threshold_weights
from .callbacks.base import Callback, CallbackList
from .callbacks.logging import LoggingCallback, EnhancedProgressBar
from .callbacks.checkpointing import CheckpointingCallback, EarlyStoppingCallback
from .callbacks.visualization import VisualizationCallback
from .optimization.lr_schedulers import create_scheduler
from .optimization.regularization import GradientClipping, CompositeRegularization, DropPathScheduler


class Trainer:
    """
    Classe principale pour l'entraînement des modèles de segmentation.
    
    Cette classe encapsule toute la logique d'entraînement, d'évaluation et
    de test des modèles, avec une configuration flexible et de nombreuses
    options d'optimisation.
    """
    
    def __init__(self, model: nn.Module, config: Config, 
                 train_loader: DataLoader, val_loader: DataLoader, 
                 test_loader: Optional[DataLoader] = None,
                 device: Optional[torch.device] = None,
                 callbacks: Optional[List[Callback]] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[Any] = None,
                 loss_fn: Optional[nn.Module] = None):
        """
        Initialise le Trainer.
        
        Args:
            model: Modèle à entraîner.
            config: Configuration d'entraînement.
            train_loader: DataLoader pour l'entraînement.
            val_loader: DataLoader pour la validation.
            test_loader: DataLoader pour le test (optionnel).
            device: Dispositif sur lequel effectuer l'entraînement (CPU/GPU).
            callbacks: Liste de callbacks pour personnaliser l'entraînement.
            optimizer: Optimiseur à utiliser (si None, créé à partir de la config).
            scheduler: Scheduler de taux d'apprentissage (si None, créé à partir de la config).
            loss_fn: Fonction de perte à utiliser (si None, créée à partir de la config).
        """
        # Paramètres de base
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device if device is not None else get_device()
        
        # Configuration de l'entraînement
        self.epochs = getattr(config, 'epochs', 100)
        self.thresholds = getattr(config, 'thresholds', [5, 10, 15, 20])
        self.model_name = getattr(config, 'model_name', model.__class__.__name__)
        
        # Répertoires
        self.log_dir = getattr(config, 'LOGS_DIR', 'logs')
        self.checkpoint_dir = getattr(config, 'CHECKPOINTS_DIR', 'checkpoints')
        
        # Métriques et critères
        self.segmentation_metrics = SegmentationMetrics(device=self.device)
        self.threshold_metrics = ThresholdMetrics(self.thresholds) if self.thresholds else None
        
        # Configurer l'optimiseur
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
        
        # Configurer la fonction de perte
        if loss_fn is None:
            self.loss_fn = self._create_loss_function()
        else:
            self.loss_fn = loss_fn
        
        # Configurer le scheduler
        if scheduler is None:
            self.scheduler = self._create_scheduler()
        else:
            self.scheduler = scheduler
        
        # Techniques d'optimisation avancées
        self.gradient_clipper = None
        self.droppath_scheduler = None
        
        if getattr(config, 'use_gradient_clipping', False):
            self.gradient_clipper = GradientClipping(
                model=model,
                clip_value=getattr(config, 'clip_value', 1.0),
                clip_norm=getattr(config, 'clip_norm', True),
                monitor=getattr(config, 'monitor_gradients', False)
            )
        
        if getattr(config, 'use_droppath', False):
            self.droppath_scheduler = DropPathScheduler(
                model=model,
                start_prob=getattr(config, 'droppath_start_prob', 0.0),
                final_prob=getattr(config, 'droppath_final_prob', 0.2),
                epochs=self.epochs,
                strategy=getattr(config, 'droppath_strategy', 'linear')
            )
        
        # Callbacks
        self.callbacks = CallbackList(callbacks if callbacks else [])
        self._setup_default_callbacks()
        
        # État interne
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.early_stop = False
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'lr': [],
            'epoch_times': []
        }
        
        # Déplacer le modèle sur le device cible
        self.model.to(self.device)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Crée l'optimiseur à partir de la configuration.
        
        Returns:
            Optimiseur configuré.
        """
        optimizer_type = getattr(self.config, 'optimizer', 'adam')
        lr = getattr(self.config, 'learning_rate', 1e-3)
        weight_decay = getattr(self.config, 'weight_decay', 1e-4)
        
        if optimizer_type.lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=getattr(self.config, 'adam_betas', (0.9, 0.999))
            )
        elif optimizer_type.lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=getattr(self.config, 'adam_betas', (0.9, 0.999))
            )
        elif optimizer_type.lower() == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=getattr(self.config, 'momentum', 0.9),
                weight_decay=weight_decay,
                nesterov=getattr(self.config, 'nesterov', True)
            )
        else:
            raise ValueError(f"Type d'optimiseur non reconnu: {optimizer_type}")
    
    def _create_loss_function(self) -> nn.Module:
        """
        Crée la fonction de perte à partir de la configuration.
        
        Returns:
            Fonction de perte configurée.
        """
        loss_type = getattr(self.config, 'loss_type', 'combined_focal_dice')
        
        if getattr(self.config, 'use_threshold_weights', False):
            return create_loss_with_threshold_weights(self.config)
        else:
            return create_loss_function(self.config)
    
    def _create_scheduler(self) -> Any:
        """
        Crée le scheduler de taux d'apprentissage à partir de la configuration.
        
        Returns:
            Scheduler configuré.
        """
        return create_scheduler(
            self.optimizer, 
            self.config, 
            len(self.train_loader)
        )
    
    def _setup_default_callbacks(self) -> None:
        """
        Configure les callbacks par défaut si aucun n'est fourni.
        """
        # Ne pas ajouter les callbacks par défaut si des callbacks personnalisés sont fournis
        if not self.callbacks.callbacks:
            # Callback de logging
            logging_callback = LoggingCallback(
                log_dir=self.log_dir,
                model_name=self.model_name,
                use_tensorboard=getattr(self.config, 'use_tensorboard', True)
            )
            self.callbacks.append(logging_callback)
            
            # Callback de sauvegarde des points de contrôle
            checkpoint_callback = CheckpointingCallback(
                checkpoint_dir=self.checkpoint_dir,
                model_name=self.model_name,
                save_best_only=getattr(self.config, 'save_best_only', True),
                monitor=getattr(self.config, 'monitor_metric', 'val_iou')
            )
            self.callbacks.append(checkpoint_callback)
            
            # Callback de visualisation si un dataset de validation est fourni
            if hasattr(self.val_loader, 'dataset'):
                visualization_callback = VisualizationCallback(
                    log_dir=self.log_dir,
                    val_dataset=self.val_loader.dataset,
                    num_samples=min(4, len(self.val_loader.dataset)),
                    save_frequency=getattr(self.config, 'vis_save_frequency', 5)
                )
                self.callbacks.append(visualization_callback)
            
            # Callback d'early stopping
            if getattr(self.config, 'use_early_stopping', False):
                early_stopping_callback = EarlyStoppingCallback(
                    monitor=getattr(self.config, 'early_stopping_monitor', 'val_loss'),
                    patience=getattr(self.config, 'early_stopping_patience', 10),
                    mode=getattr(self.config, 'early_stopping_mode', 'min')
                )
                self.callbacks.append(early_stopping_callback)
            
            # Barre de progression
            if getattr(self.config, 'use_progress_bar', True):
                progress_bar = EnhancedProgressBar(
                    total_epochs=self.epochs,
                    steps_per_epoch=len(self.train_loader)
                )
                self.callbacks.append(progress_bar)
    
    def train(self) -> Dict[str, Any]:
        """
        Entraîne le modèle pour le nombre d'époques spécifié.
        
        Returns:
            Historique d'entraînement.
        """
        # Informer les callbacks du début de l'entraînement
        self.callbacks.on_train_begin({
            'config': vars(self.config) if hasattr(self.config, '__dict__') else self.config,
            'model': self.model,
            'model_summary': str(self.model)
        })
        
        # Boucle principale d'entraînement
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            
            # Informer les callbacks du début de l'époque
            self.callbacks.on_epoch_begin(epoch, {})
            
            # Mettre à jour le taux de DropPath si utilisé
            if self.droppath_scheduler:
                self.droppath_scheduler.step(epoch)
            
            # Mesurer le temps d'époque
            epoch_start_time = time.time()
            
            # Entraînement pour cette époque
            train_metrics = self._train_epoch(epoch)
            
            # Validation pour cette époque
            val_metrics = self._validate_epoch(epoch)
            
            # Calculer le temps d'époque
            epoch_time = time.time() - epoch_start_time
            
            # Mettre à jour le scheduler si nécessaire
            if self.scheduler:
                # Vérifier le type de scheduler pour déterminer s'il faut passer une métrique
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau nécessite une métrique
                    monitor_metric = getattr(self.config, 'monitor_metric', 'val_loss')
                    metric_parts = monitor_metric.split('_', 1)
                    if len(metric_parts) == 2 and metric_parts[0] == 'val' and metric_parts[1] in val_metrics:
                        metric_value = val_metrics[metric_parts[1]]
                        self.scheduler.step(metric_value)
                    else:
                        # Utiliser val_loss par défaut
                        self.scheduler.step(val_metrics.get('loss', 0))
                else:
                    # Autres schedulers ne nécessitent pas de métrique
                    self.scheduler.step()
            
            # Récupérer le learning rate actuel
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Mettre à jour l'historique
            self.history['train_loss'].append(train_metrics.get('loss', 0))
            self.history['val_loss'].append(val_metrics.get('loss', 0))
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['lr'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)
            
            # Vérifier si c'est le meilleur modèle
            improved = False
            monitor_metric = getattr(self.config, 'monitor_metric', 'val_iou')
            metric_parts = monitor_metric.split('_', 1)
            
            if len(metric_parts) == 2 and metric_parts[0] == 'val' and metric_parts[1] in val_metrics:
                current_metric = val_metrics[metric_parts[1]]
                
                if (metric_parts[1] == 'loss' and current_metric < self.best_metric) or \
                   (metric_parts[1] != 'loss' and current_metric > self.best_metric):
                    self.best_metric = current_metric
                    improved = True
            
            # Informer les callbacks de la fin de l'époque
            self.callbacks.on_epoch_end(epoch, {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'lr': current_lr,
                'improved': improved,
                'model': self.model,
                'optimizer': self.optimizer,
                'epoch_time': epoch_time,
                'device': self.device
            })
            
            # Vérifier si l'entraînement doit être arrêté prématurément
            if self.early_stop:
                print(f"Arrêt anticipé à l'époque {epoch+1}")
                break
        
        # Informer les callbacks de la fin de l'entraînement
        total_time = time.time() - start_time
        
        # Calculer les meilleures métriques
        best_metrics = {
            'best_' + getattr(self.config, 'monitor_metric', 'val_iou'): self.best_metric,
            'total_time': total_time,
            'epochs_completed': self.current_epoch + 1
        }
        
        self.callbacks.on_train_end({
            'history': self.history,
            'best_metrics': best_metrics
        })
        
        return self.history
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Effectue une époque d'entraînement.
        
        Args:
            epoch: Numéro de l'époque actuelle.
            
        Returns:
            Métriques d'entraînement.
        """
        # Mettre le modèle en mode entraînement
        self.model.train()
        
        # Réinitialiser les métriques
        self.segmentation_metrics.reset()
        if self.threshold_metrics:
            self.threshold_metrics.reset()
        
        # Compteurs pour le suivi
        total_loss = 0.0
        n_batches = len(self.train_loader)
        
        # Informer les callbacks du début de la boucle d'entraînement
        self.callbacks.on_validation_begin({})
        
        # Boucle sur les batches
        for batch_idx, batch in enumerate(self.train_loader):
            # Informer les callbacks du début du batch
            self.callbacks.on_batch_begin(batch_idx, {'epoch': epoch})
            
            # Extraire les données du batch
            if isinstance(batch, dict):
                dsm = batch['dsm'].to(self.device)
                mask = batch['mask'].to(self.device)
                threshold = batch.get('threshold', torch.tensor([10.0])).to(self.device)
            else:
                dsm, mask = batch[0].to(self.device), batch[1].to(self.device)
                threshold = batch[2].to(self.device) if len(batch) > 2 else torch.tensor([10.0], device=self.device)
            
            # Réinitialiser les gradients
            self.optimizer.zero_grad()
            
            # Passe avant
            outputs = self.model(dsm, threshold)
            
            # Calcul de la perte
            loss = self.loss_fn(outputs, mask, threshold)
            
            # Rétropropagation
            loss.backward()
            
            # Appliquer le gradient clipping si configuré
            if self.gradient_clipper:
                self.gradient_clipper.clip_gradients()
            
            # Mettre à jour les poids
            self.optimizer.step()
            
            # Mettre à jour les métriques
            batch_size = dsm.size(0)
            total_loss += loss.item() * batch_size
            
            with torch.no_grad():
                batch_metrics = self.segmentation_metrics.update(outputs, mask)
                
                # Mettre à jour les métriques par seuil si configuré
                if self.threshold_metrics:
                    for i in range(threshold.size(0)):
                        t_val = threshold[i].item()
                        self.threshold_metrics.update(
                            outputs[i:i+1], mask[i:i+1], 0.5, t_val
                        )
            
            # Informer les callbacks de la fin du batch
            self.callbacks.on_batch_end(batch_idx, {
                'epoch': epoch,
                'metrics': {**batch_metrics, 'loss': loss.item()},
                'batch_size': batch_size
            })
        
        # Calculer les métriques finales pour l'époque
        metrics = self.segmentation_metrics.compute()
        metrics['loss'] = total_loss / len(self.train_loader.dataset)
        
        # Ajouter les métriques par seuil si configuré
        if self.threshold_metrics:
            metrics['threshold_metrics'] = self.threshold_metrics.compute()
        
        # Informer les callbacks de la fin de la boucle d'entraînement
        self.callbacks.on_validation_end({})
        
        return metrics
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Effectue une époque de validation.
        
        Args:
            epoch: Numéro de l'époque actuelle.
            
        Returns:
            Métriques de validation.
        """
        # Mettre le modèle en mode évaluation
        self.model.eval()
        
        # Réinitialiser les métriques
        self.segmentation_metrics.reset()
        if self.threshold_metrics:
            self.threshold_metrics.reset()
        
        # Compteurs pour le suivi
        total_loss = 0.0
        
        # Informer les callbacks du début de la validation
        self.callbacks.on_validation_begin({})
        
        # Boucle sur les batches
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Informer les callbacks du début du batch
                self.callbacks.on_validation_batch_begin(batch_idx, {'epoch': epoch})
                
                # Extraire les données du batch
                if isinstance(batch, dict):
                    dsm = batch['dsm'].to(self.device)
                    mask = batch['mask'].to(self.device)
                    threshold = batch.get('threshold', torch.tensor([10.0])).to(self.device)
                else:
                    dsm, mask = batch[0].to(self.device), batch[1].to(self.device)
                    threshold = batch[2].to(self.device) if len(batch) > 2 else torch.tensor([10.0], device=self.device)
                
                # Passe avant
                outputs = self.model(dsm, threshold)
                
                # Calcul de la perte
                loss = self.loss_fn(outputs, mask, threshold)
                
                # Mettre à jour les métriques
                batch_size = dsm.size(0)
                total_loss += loss.item() * batch_size
                
                # Mise à jour des métriques
                self.segmentation_metrics.update(outputs, mask)
                
                # Mettre à jour les métriques par seuil si configuré
                if self.threshold_metrics:
                    for i in range(threshold.size(0)):
                        t_val = threshold[i].item()
                        self.threshold_metrics.update(
                            outputs[i:i+1], mask[i:i+1], 0.5, t_val
                        )
                
                # Informer les callbacks de la fin du batch
                self.callbacks.on_validation_batch_end(batch_idx, {'epoch': epoch})
        
        # Calculer les métriques finales pour l'époque
        metrics = self.segmentation_metrics.compute()
        metrics['loss'] = total_loss / len(self.val_loader.dataset)
        
        # Calculer la matrice de confusion
        confusion_matrix = self.segmentation_metrics.compute_confusion_matrix()
        metrics['confusion_matrix'] = confusion_matrix
        
        # Ajouter les métriques par seuil si configuré
        if self.threshold_metrics:
            metrics['threshold_metrics'] = self.threshold_metrics.compute()
        
        # Informer les callbacks de la fin de la validation
        self.callbacks.on_validation_end({})
        
        return metrics
    
    def test(self) -> Dict[str, Any]:
        """
        Teste le modèle sur le jeu de test.
        
        Returns:
            Métriques de test.
        """
        if self.test_loader is None:
            raise ValueError("Aucun DataLoader de test n'a été fourni")
        
        # Mettre le modèle en mode évaluation
        self.model.eval()
        
        # Réinitialiser les métriques
        self.segmentation_metrics.reset()
        if self.threshold_metrics:
            self.threshold_metrics.reset()
        
        # Compteurs pour le suivi
        total_loss = 0.0
        
        # Informer les callbacks du début du test
        self.callbacks.on_test_begin({})
        
        # Boucle sur les batches
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # Informer les callbacks du début du batch
                self.callbacks.on_test_batch_begin(batch_idx, {})
                
                # Extraire les données du batch
                if isinstance(batch, dict):
                    dsm = batch['dsm'].to(self.device)
                    mask = batch['mask'].to(self.device)
                    threshold = batch.get('threshold', torch.tensor([10.0])).to(self.device)
                else:
                    dsm, mask = batch[0].to(self.device), batch[1].to(self.device)
                    threshold = batch[2].to(self.device) if len(batch) > 2 else torch.tensor([10.0], device=self.device)
                
                # Passe avant
                outputs = self.model(dsm, threshold)
                
                # Calcul de la perte
                loss = self.loss_fn(outputs, mask, threshold)
                
                # Mettre à jour les métriques
                batch_size = dsm.size(0)
                total_loss += loss.item() * batch_size
                
                # Mise à jour des métriques
                self.segmentation_metrics.update(outputs, mask)
                
                # Mettre à jour les métriques par seuil si configuré
                if self.threshold_metrics:
                    for i in range(threshold.size(0)):
                        t_val = threshold[i].item()
                        self.threshold_metrics.update(
                            outputs[i:i+1], mask[i:i+1], 0.5, t_val
                        )
                
                # Informer les callbacks de la fin du batch
                self.callbacks.on_test_batch_end(batch_idx, {})
        
        # Calculer les métriques finales pour le test
        metrics = self.segmentation_metrics.compute()
        metrics['loss'] = total_loss / len(self.test_loader.dataset)
        
        # Calculer la matrice de confusion
        confusion_matrix = self.segmentation_metrics.compute_confusion_matrix()
        metrics['confusion_matrix'] = confusion_matrix
        
        # Ajouter les métriques par seuil si configuré
        if self.threshold_metrics:
            metrics['threshold_metrics'] = self.threshold_metrics.compute()
        
        # Informer les callbacks de la fin du test
        self.callbacks.on_test_end({
            'metrics': metrics,
            'model': self.model,
            'device': self.device
        })
        
        return metrics
    
    def predict(self, inputs: torch.Tensor, thresholds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Effectue des prédictions sur des données d'entrée.
        
        Args:
            inputs: Tensor d'entrée.
            thresholds: Tensor des seuils de hauteur.
            
        Returns:
            Prédictions du modèle.
        """
        # Vérifier les arguments
        if thresholds is None:
            # Utiliser 10.0 comme seuil par défaut
            thresholds = torch.tensor([10.0] * inputs.size(0), device=self.device)
        
        # Mettre le modèle en mode évaluation
        self.model.eval()
        
        # Déplacer les données sur le dispositif cible
        inputs = inputs.to(self.device)
        thresholds = thresholds.to(self.device)
        
        # Effectuer la prédiction
        with torch.no_grad():
            outputs = self.model(inputs, thresholds)
        
        return outputs
    
    def save_checkpoint(self, filepath: Optional[str] = None) -> str:
        """
        Sauvegarde un point de contrôle du modèle.
        
        Args:
            filepath: Chemin où sauvegarder le point de contrôle.
            
        Returns:
            Chemin du fichier de sauvegarde.
        """
        # Générer un nom de fichier par défaut si non spécifié
        if filepath is None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            filepath = os.path.join(
                self.checkpoint_dir, 
                f"{self.model_name}_epoch_{self.current_epoch:03d}.pt"
            )
        
        # Préparer les données à sauvegarder
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'history': self.history,
            'thresholds': self.thresholds,
            'config': vars(self.config) if hasattr(self.config, '__dict__') else self.config
        }
        
        # Ajouter l'état du scheduler si existant
        if self.scheduler is not None and hasattr(self.scheduler, 'state_dict'):
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Sauvegarder le point de contrôle
        torch.save(checkpoint, filepath)
        
        return filepath
    
    def load_checkpoint(self, filepath: str, strict: bool = True) -> None:
        """
        Charge un point de contrôle du modèle.
        
        Args:
            filepath: Chemin du fichier de point de contrôle.
            strict: Charger strictement les poids (True) ou ignorer les paramètres manquants (False).
        """
        # Vérifier que le fichier existe
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Le fichier de point de contrôle {filepath} n'existe pas.")
        
        # Charger le point de contrôle
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Charger l'état du modèle
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Charger l'état de l'optimiseur si disponible
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Charger l'état du scheduler si disponible
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None and hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restaurer l'état interne
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.best_metric = checkpoint.get('best_metric', float('-inf'))
        
        # Restaurer l'historique si disponible
        if 'history' in checkpoint:
            self.history = checkpoint['history']
    
    def resume_training(self, filepath: str) -> Dict[str, Any]:
        """
        Reprend l'entraînement à partir d'un point de contrôle.
        
        Args:
            filepath: Chemin du fichier de point de contrôle.
            
        Returns:
            Historique d'entraînement mis à jour.
        """
        # Charger le point de contrôle
        self.load_checkpoint(filepath)
        
        # Reprendre l'entraînement
        return self.train()


def train_model(model: nn.Module, config: Config, 
               train_loader: DataLoader, val_loader: DataLoader, 
               test_loader: Optional[DataLoader] = None,
               device: Optional[torch.device] = None,
               callbacks: Optional[List[Callback]] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Fonction utilitaire pour entraîner un modèle avec la classe Trainer.
    
    Args:
        model: Modèle à entraîner.
        config: Configuration d'entraînement.
        train_loader: DataLoader pour l'entraînement.
        val_loader: DataLoader pour la validation.
        test_loader: DataLoader pour le test (optionnel).
        device: Dispositif sur lequel effectuer l'entraînement (CPU/GPU).
        callbacks: Liste de callbacks pour personnaliser l'entraînement.
        
    Returns:
        Tuple contenant le modèle entraîné et l'historique d'entraînement.
    """
    # Créer le trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        callbacks=callbacks
    )
    
    # Entraîner le modèle
    history = trainer.train()
    
    # Tester le modèle si un loader de test est fourni
    if test_loader is not None:
        test_metrics = trainer.test()
        print("Métriques de test:")
        for metric, value in test_metrics.items():
            if isinstance(value, dict):
                print(f"  {metric}:")
                for sub_metric, sub_value in value.items():
                    print(f"    {sub_metric}: {sub_value}")
            else:
                print(f"  {metric}: {value}")
    
    return model, history 