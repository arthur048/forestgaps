"""
Module de callbacks de logging pour l'entraînement.

Ce module fournit des callbacks pour le logging des métriques et des informations
pendant l'entraînement des modèles.
"""

import os
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from .base import Callback


class LoggingCallback(Callback):
    """
    Callback pour le logging des métriques d'entraînement.
    
    Ce callback gère le logging des métriques d'entraînement dans la console,
    dans des fichiers de log et éventuellement dans TensorBoard.
    """
    
    def __init__(self, log_dir: str, model_name: Optional[str] = None, 
                 use_tensorboard: bool = True, log_frequency: int = 10,
                 metrics_to_track: Optional[List[str]] = None):
        """
        Initialise le callback de logging.
        
        Args:
            log_dir: Répertoire où sauvegarder les logs.
            model_name: Nom du modèle pour la génération des logs.
            use_tensorboard: Utiliser TensorBoard pour la visualisation.
            log_frequency: Fréquence de logging des métriques (en batches).
            metrics_to_track: Liste des métriques à suivre. Si None, toutes les métriques sont suivies.
        """
        super(LoggingCallback, self).__init__()
        
        # Paramètres de base
        self.log_dir = log_dir
        self.model_name = model_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.use_tensorboard = use_tensorboard
        self.log_frequency = log_frequency
        self.metrics_to_track = metrics_to_track or ['loss', 'iou', 'f1', 'precision', 'recall']
        
        # Créer le répertoire de logs s'il n'existe pas
        os.makedirs(log_dir, exist_ok=True)
        
        # Configurer le logger
        self.log_file = os.path.join(log_dir, f"{self.model_name}_training.log")
        self.logger = self._setup_logger()
        
        # Initialiser TensorBoard si nécessaire
        self.writer = None
        if use_tensorboard:
            self.writer = SummaryWriter(os.path.join(log_dir, 'tensorboard', self.model_name))
        
        # Métriques et historiques
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'lr': [],
            'epoch_times': []
        }
        
        # Timestamps pour le suivi des performances
        self.epoch_start_time = None
        self.train_start_time = None
    
    def _setup_logger(self) -> logging.Logger:
        """
        Configure le logger pour le callback.
        
        Returns:
            Logger configuré.
        """
        logger = logging.getLogger(f"{self.model_name}_training")
        logger.setLevel(logging.INFO)
        
        # Gérer les handlers existants
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Créer un handler pour la console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # Créer un handler pour le fichier de log
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        return logger
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé au début de l'entraînement.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        self.train_start_time = time.time()
        
        # Journaliser le début de l'entraînement
        self.logger.info(f"Début de l'entraînement du modèle {self.model_name}")
        
        # Journaliser la configuration si disponible
        if logs and 'config' in logs:
            config_str = json.dumps(logs['config'], indent=2)
            self.logger.info(f"Configuration:\n{config_str}")
            
            if self.writer:
                self.writer.add_text('Configuration', config_str)
        
        # Journaliser l'architecture du modèle si disponible
        if logs and 'model_summary' in logs:
            self.logger.info(f"Architecture du modèle:\n{logs['model_summary']}")
            
            if self.writer and 'model' in logs:
                try:
                    # Ajouter le graphe du modèle à TensorBoard
                    dummy_input = logs.get('dummy_input')
                    if dummy_input is not None:
                        self.writer.add_graph(logs['model'], dummy_input)
                except Exception as e:
                    self.logger.warning(f"Impossible d'ajouter le graphe du modèle à TensorBoard: {e}")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de l'entraînement.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        total_time = time.time() - self.train_start_time
        
        # Journaliser la fin de l'entraînement
        self.logger.info(f"Fin de l'entraînement du modèle {self.model_name}")
        self.logger.info(f"Temps total d'entraînement: {total_time:.2f} secondes")
        
        # Journaliser les meilleures métriques
        if logs and 'best_metrics' in logs:
            best_metrics = logs['best_metrics']
            self.logger.info(f"Meilleures métriques: {best_metrics}")
            
            if self.writer:
                for metric_name, metric_value in best_metrics.items():
                    self.writer.add_scalar(f'Best/{metric_name}', metric_value, 0)
        
        # Sauvegarder l'historique des métriques
        history_file = os.path.join(self.log_dir, f"{self.model_name}_history.json")
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Tracer l'évolution des métriques
        if len(self.history['train_loss']) > 0:
            self._plot_metrics()
        
        # Fermer TensorBoard writer
        if self.writer:
            self.writer.close()
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé au début de chaque époque.
        
        Args:
            epoch: Numéro de l'époque actuelle.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        self.epoch_start_time = time.time()
        self.logger.info(f"Début de l'époque {epoch+1}")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de chaque époque.
        
        Args:
            epoch: Numéro de l'époque actuelle.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        epoch_time = time.time() - self.epoch_start_time
        self.history['epoch_times'].append(epoch_time)
        
        # Extraire les métriques
        logs = logs or {}
        train_metrics = logs.get('train_metrics', {})
        val_metrics = logs.get('val_metrics', {})
        learning_rate = logs.get('lr', 0.0)
        
        # Mettre à jour l'historique
        self.history['train_loss'].append(train_metrics.get('loss', 0.0))
        self.history['val_loss'].append(val_metrics.get('loss', 0.0))
        self.history['train_metrics'].append(train_metrics)
        self.history['val_metrics'].append(val_metrics)
        self.history['lr'].append(learning_rate)
        
        # Journaliser les métriques
        log_str = f"Époque {epoch+1} terminée en {epoch_time:.2f}s - "
        metrics_str = []
        
        # Métriques d'entraînement
        train_str = []
        for metric in self.metrics_to_track:
            if metric in train_metrics:
                train_str.append(f"{metric}: {train_metrics[metric]:.4f}")
        if train_str:
            metrics_str.append("Train: " + ", ".join(train_str))
        
        # Métriques de validation
        val_str = []
        for metric in self.metrics_to_track:
            if metric in val_metrics:
                val_str.append(f"{metric}: {val_metrics[metric]:.4f}")
        if val_str:
            metrics_str.append("Val: " + ", ".join(val_str))
        
        # Taux d'apprentissage
        metrics_str.append(f"LR: {learning_rate:.6f}")
        
        # Journaliser
        log_str += " | ".join(metrics_str)
        self.logger.info(log_str)
        
        # Ajouter à TensorBoard
        if self.writer:
            # Métriques d'entraînement
            for metric_name, metric_value in train_metrics.items():
                self.writer.add_scalar(f'Train/{metric_name}', metric_value, epoch)
            
            # Métriques de validation
            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
            
            # Taux d'apprentissage
            self.writer.add_scalar('Train/learning_rate', learning_rate, epoch)
            
            # Temps d'époque
            self.writer.add_scalar('Train/epoch_time', epoch_time, epoch)
            
            # Ajouter des métriques par seuil si disponibles
            if 'threshold_metrics' in val_metrics:
                for threshold, metrics in val_metrics['threshold_metrics'].items():
                    for metric_name, metric_value in metrics.items():
                        self.writer.add_scalar(f'Threshold/{threshold}/{metric_name}', 
                                              metric_value, epoch)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de chaque batch.
        
        Args:
            batch: Numéro du batch actuel.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        if batch % self.log_frequency == 0 and logs:
            # Extraire les métriques
            metrics = logs.get('metrics', {})
            epoch = logs.get('epoch', 0)
            
            # Journaliser
            log_str = f"Époque {epoch+1}, Batch {batch} - "
            metrics_str = []
            
            for metric in self.metrics_to_track:
                if metric in metrics:
                    metrics_str.append(f"{metric}: {metrics[metric]:.4f}")
            
            if metrics_str:
                log_str += " | ".join(metrics_str)
                self.logger.debug(log_str)
    
    def _plot_metrics(self) -> None:
        """
        Trace l'évolution des métriques au cours de l'entraînement.
        """
        # Récupérer les données
        epochs = range(1, len(self.history['train_loss']) + 1)
        train_loss = self.history['train_loss']
        val_loss = self.history['val_loss']
        
        # Créer la figure
        plt.figure(figsize=(15, 10))
        
        # Sous-figure des pertes
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_loss, 'b-', label='Train Loss')
        plt.plot(epochs, val_loss, 'r-', label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Sous-figure pour le taux d'apprentissage
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.history['lr'], 'g-')
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        
        # Sous-figure pour les métriques d'entraînement
        plt.subplot(2, 2, 3)
        for metric in ['iou', 'f1', 'precision', 'recall']:
            values = [metrics.get(metric, 0.0) for metrics in self.history['train_metrics']]
            plt.plot(epochs, values, label=metric.capitalize())
        plt.title('Training Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        
        # Sous-figure pour les métriques de validation
        plt.subplot(2, 2, 4)
        for metric in ['iou', 'f1', 'precision', 'recall']:
            values = [metrics.get(metric, 0.0) for metrics in self.history['val_metrics']]
            plt.plot(epochs, values, label=metric.capitalize())
        plt.title('Validation Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder la figure
        plot_path = os.path.join(self.log_dir, f"{self.model_name}_metrics.png")
        plt.savefig(plot_path)
        plt.close()


class EnhancedProgressBar(Callback):
    """
    Callback pour afficher une barre de progression améliorée pendant l'entraînement.
    
    Cette barre de progression affiche des métriques en temps réel et s'adapte
    à l'environnement d'exécution (Google Colab ou terminal).
    """
    
    def __init__(self, total_epochs: int, steps_per_epoch: int, 
                 metrics_to_display: Optional[List[str]] = None,
                 update_interval: int = 10):
        """
        Initialise la barre de progression.
        
        Args:
            total_epochs: Nombre total d'époques.
            steps_per_epoch: Nombre de batches par époque.
            metrics_to_display: Liste des métriques à afficher.
            update_interval: Intervalle de mise à jour de la barre (en batches).
        """
        super(EnhancedProgressBar, self).__init__()
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.update_interval = update_interval
        self.metrics_to_display = metrics_to_display or ['loss', 'iou', 'f1']
        
        self.current_epoch = 0
        self.current_step = 0
        self.epoch_start_time = None
        
        # Déterminer si nous sommes dans Colab
        self.is_colab = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False
        
        # Configurer tqdm si disponible
        self.use_tqdm = False
        try:
            from tqdm.auto import tqdm
            self.tqdm = tqdm
            self.use_tqdm = True
        except ImportError:
            pass
        
        self.pbar = None
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé au début de chaque époque.
        
        Args:
            epoch: Numéro de l'époque actuelle.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        self.current_epoch = epoch
        self.current_step = 0
        self.epoch_start_time = time.time()
        
        # Créer la barre de progression
        if self.use_tqdm:
            self.pbar = self.tqdm(total=self.steps_per_epoch, desc=f"Époque {epoch+1}/{self.total_epochs}")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de chaque époque.
        
        Args:
            epoch: Numéro de l'époque actuelle.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        if self.pbar:
            self.pbar.close()
            self.pbar = None
        
        # Calculer le temps écoulé
        epoch_time = time.time() - self.epoch_start_time
        
        # Extraire les métriques
        logs = logs or {}
        train_metrics = logs.get('train_metrics', {})
        val_metrics = logs.get('val_metrics', {})
        
        # Afficher un résumé de l'époque
        metrics_str = []
        
        # Métriques d'entraînement
        train_str = []
        for metric in self.metrics_to_display:
            if metric in train_metrics:
                train_str.append(f"{metric}: {train_metrics[metric]:.4f}")
        if train_str:
            metrics_str.append("Train: " + ", ".join(train_str))
        
        # Métriques de validation
        val_str = []
        for metric in self.metrics_to_display:
            if metric in val_metrics:
                val_str.append(f"{metric}: {val_metrics[metric]:.4f}")
        if val_str:
            metrics_str.append("Val: " + ", ".join(val_str))
        
        # Temps d'époque
        metrics_str.append(f"Time: {epoch_time:.2f}s")
        
        # Afficher le résumé
        if self.is_colab:
            from IPython.display import display, clear_output
            clear_output(wait=True)
            print(f"Époque {epoch+1}/{self.total_epochs} - " + " | ".join(metrics_str))
        else:
            print(f"\nÉpoque {epoch+1}/{self.total_epochs} - " + " | ".join(metrics_str))
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de chaque batch.
        
        Args:
            batch: Numéro du batch actuel.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        self.current_step = batch
        
        # Mettre à jour la barre de progression
        if self.pbar and batch % self.update_interval == 0:
            # Extraire les métriques
            metrics = logs.get('metrics', {}) if logs else {}
            
            # Mettre à jour la description
            postfix = {}
            for metric in self.metrics_to_display:
                if metric in metrics:
                    postfix[metric] = f"{metrics[metric]:.4f}"
            
            self.pbar.set_postfix(postfix)
            self.pbar.update(self.update_interval) 