"""
Module de callbacks de checkpointing pour l'entraînement.

Ce module fournit des callbacks pour la sauvegarde des points de contrôle
pendant l'entraînement des modèles.
"""

import os
import json
import time
import shutil
from typing import Dict, Any, Optional, List, Union, Callable

import torch
import numpy as np

from .base import Callback


class CheckpointingCallback(Callback):
    """
    Callback pour la sauvegarde des points de contrôle pendant l'entraînement.
    
    Ce callback permet de sauvegarder le modèle, l'optimiseur et d'autres 
    informations d'état à différents moments de l'entraînement.
    """
    
    def __init__(self, checkpoint_dir: str, model_name: Optional[str] = None,
                 save_best_only: bool = True, save_weights_only: bool = False,
                 monitor: str = 'val_iou', mode: str = 'max',
                 save_frequency: Optional[int] = None, max_to_keep: int = 3,
                 include_optimizer: bool = True, verbose: bool = True):
        """
        Initialise le callback de checkpointing.
        
        Args:
            checkpoint_dir: Répertoire où sauvegarder les points de contrôle.
            model_name: Nom du modèle pour les fichiers de sauvegarde.
            save_best_only: Ne sauvegarder que le meilleur modèle selon la métrique.
            save_weights_only: Ne sauvegarder que les poids du modèle.
            monitor: Métrique à surveiller pour déterminer le meilleur modèle.
            mode: Mode de comparaison ('min' ou 'max').
            save_frequency: Fréquence de sauvegarde (en époques). Si None, sauvegarde à chaque époque.
            max_to_keep: Nombre maximum de points de contrôle à conserver.
            include_optimizer: Inclure l'état de l'optimiseur dans la sauvegarde.
            verbose: Afficher des informations lors des sauvegardes.
        """
        super(CheckpointingCallback, self).__init__()
        
        # Paramètres de base
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name or "model"
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.monitor = monitor
        self.mode = mode
        self.save_frequency = save_frequency
        self.max_to_keep = max_to_keep
        self.include_optimizer = include_optimizer
        self.verbose = verbose
        
        # Créer le répertoire de checkpoints s'il n'existe pas
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Déterminer le mode de comparaison
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode '{mode}' non reconnu. Utilisez 'min' ou 'max'.")
        
        # Initialiser les valeurs
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.checkpoints = []  # Liste des chemins des points de contrôle
        self.last_epoch_saved = None
    
    def _compare_metric(self, current: float, best: float) -> bool:
        """
        Compare la métrique actuelle avec la meilleure valeur.
        
        Args:
            current: Valeur actuelle de la métrique.
            best: Meilleure valeur précédente.
            
        Returns:
            True si la métrique actuelle est meilleure, False sinon.
        """
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def _get_monitored_value(self, logs: Dict[str, Any]) -> Optional[float]:
        """
        Récupère la valeur de la métrique surveillée.
        
        Args:
            logs: Dictionnaire contenant les métriques.
            
        Returns:
            Valeur de la métrique surveillée, ou None si non disponible.
        """
        # Extraire le nom de la métrique et le préfixe
        metric_parts = self.monitor.split('_', 1)
        
        if len(metric_parts) == 2:
            prefix, metric_name = metric_parts
            
            # Chercher dans les métriques spécifiques à l'étape (train/val)
            if prefix in logs and metric_name in logs[prefix]:
                return logs[prefix][metric_name]
        
        # Rechercher directement dans les logs
        return logs.get(self.monitor, None)
    
    def _save_checkpoint(self, model, optimizer, epoch: int, logs: Dict[str, Any],
                         is_best: bool = False) -> str:
        """
        Sauvegarde un point de contrôle.
        
        Args:
            model: Modèle à sauvegarder.
            optimizer: Optimiseur à sauvegarder.
            epoch: Numéro de l'époque actuelle.
            logs: Dictionnaire contenant les métriques.
            is_best: Indique si c'est le meilleur modèle.
            
        Returns:
            Chemin du fichier de sauvegarde.
        """
        # Générer le nom du fichier
        monitored_value = self._get_monitored_value(logs)
        metric_str = ""
        if monitored_value is not None:
            metric_str = f"_{self.monitor}_{monitored_value:.4f}"
        
        filename = f"{self.model_name}_epoch_{epoch:03d}{metric_str}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Préparer les données à sauvegarder
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': logs,
            'timestamp': time.time()
        }
        
        # Inclure l'état de l'optimiseur si demandé
        if self.include_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Sauvegarder le checkpoint
        if self.save_weights_only:
            torch.save(model.state_dict(), filepath)
        else:
            torch.save(checkpoint, filepath)
        
        # Garder une trace des checkpoints sauvegardés
        self.checkpoints.append(filepath)
        
        # Limiter le nombre de checkpoints
        if len(self.checkpoints) > self.max_to_keep:
            oldest_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint) and not "best" in oldest_checkpoint:
                os.remove(oldest_checkpoint)
        
        # Sauvegarder une copie comme meilleur modèle si demandé
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")
            shutil.copy2(filepath, best_path)
            
            # Sauvegarder également les métriques du meilleur modèle
            best_metrics_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best_metrics.json")
            with open(best_metrics_path, 'w') as f:
                json.dump(logs, f, indent=2)
        
        return filepath
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de chaque époque.
        
        Args:
            epoch: Numéro de l'époque actuelle.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        if logs is None:
            return
        
        # Récupérer le modèle et l'optimiseur
        model = logs.get('model')
        optimizer = logs.get('optimizer')
        
        if model is None:
            return
        
        # Vérifier s'il faut sauvegarder à cette époque
        save_now = True
        
        if self.save_frequency is not None:
            save_now = (epoch % self.save_frequency == 0)
        
        # Vérifier si c'est le meilleur modèle
        is_best = False
        current_value = self._get_monitored_value(logs)
        
        if current_value is not None and self._compare_metric(current_value, self.best_value):
            self.best_value = current_value
            is_best = True
        
        # Sauvegarder le checkpoint
        if (is_best and self.save_best_only) or (save_now and not self.save_best_only):
            filepath = self._save_checkpoint(model, optimizer, epoch, logs, is_best)
            self.last_epoch_saved = epoch
            
            if self.verbose:
                message = f"Checkpoint sauvegardé: {filepath}"
                if is_best:
                    message += f" (meilleur {self.monitor}: {current_value:.4f})"
                print(message)


class EarlyStoppingCallback(Callback):
    """
    Callback pour arrêter l'entraînement si une métrique ne s'améliore plus.
    
    Ce callback surveille une métrique et arrête l'entraînement si elle ne
    s'améliore pas pendant un certain nombre d'époques.
    """
    
    def __init__(self, monitor: str = 'val_loss', mode: str = 'min',
                 patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        """
        Initialise le callback d'early stopping.
        
        Args:
            monitor: Métrique à surveiller.
            mode: Mode de comparaison ('min' ou 'max').
            patience: Nombre d'époques à attendre avant d'arrêter.
            min_delta: Amélioration minimale requise.
            verbose: Afficher des informations.
        """
        super(EarlyStoppingCallback, self).__init__()
        
        # Paramètres de base
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.mode = mode
        
        # Déterminer le mode de comparaison
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode '{mode}' non reconnu. Utilisez 'min' ou 'max'.")
        
        # Initialiser les valeurs
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
    
    def _compare_metric(self, current: float, best: float) -> bool:
        """
        Compare la métrique actuelle avec la meilleure valeur.
        
        Args:
            current: Valeur actuelle de la métrique.
            best: Meilleure valeur précédente.
            
        Returns:
            True si la métrique actuelle est meilleure, False sinon.
        """
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def _get_monitored_value(self, logs: Dict[str, Any]) -> Optional[float]:
        """
        Récupère la valeur de la métrique surveillée.
        
        Args:
            logs: Dictionnaire contenant les métriques.
            
        Returns:
            Valeur de la métrique surveillée, ou None si non disponible.
        """
        # Extraire le nom de la métrique et le préfixe
        metric_parts = self.monitor.split('_', 1)
        
        if len(metric_parts) == 2:
            prefix, metric_name = metric_parts
            
            # Chercher dans les métriques spécifiques à l'étape (train/val)
            if prefix in logs and metric_name in logs[prefix]:
                return logs[prefix][metric_name]
        
        # Rechercher directement dans les logs
        return logs.get(self.monitor, None)
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé au début de l'entraînement.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.stop_training = False
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de chaque époque.
        
        Args:
            epoch: Numéro de l'époque actuelle.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        if logs is None:
            return
        
        # Récupérer la valeur de la métrique
        current_value = self._get_monitored_value(logs)
        
        if current_value is None:
            return
        
        # Vérifier si la métrique s'améliore
        if self._compare_metric(current_value, self.best_value):
            # Réinitialiser le compteur et mettre à jour la meilleure valeur
            self.best_value = current_value
            self.wait = 0
        else:
            # Incrémenter le compteur
            self.wait += 1
            if self.wait >= self.patience:
                # Arrêter l'entraînement
                self.stopped_epoch = epoch
                self.stop_training = True
                
                # Mettre à jour les logs pour arrêter l'entraînement
                logs['stop_training'] = True
                
                if self.verbose:
                    print(f"Early stopping à l'époque {epoch+1}")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de l'entraînement.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        if self.stopped_epoch > 0 and self.verbose:
            print(f"Entraînement arrêté à l'époque {self.stopped_epoch+1}")


class ModelCheckpointHandler:
    """
    Gestionnaire pour charger et manipuler les points de contrôle.
    
    Cette classe fournit des méthodes pour charger des modèles à partir de points
    de contrôle et pour gérer les fichiers de points de contrôle.
    """
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str, model=None, map_location=None,
                        optimizer=None, strict: bool = True) -> Dict[str, Any]:
        """
        Charge un point de contrôle.
        
        Args:
            checkpoint_path: Chemin vers le fichier de point de contrôle.
            model: Modèle à charger (optionnel).
            map_location: Emplacement pour charger les tenseurs (optionnel).
            optimizer: Optimiseur à charger (optionnel).
            strict: Charger strictement les poids (True) ou ignorer les paramètres manquants (False).
            
        Returns:
            Dictionnaire contenant les données du point de contrôle.
        """
        # Vérifier que le fichier existe
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Le fichier de point de contrôle {checkpoint_path} n'existe pas.")
        
        # Charger le checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Charger l'état du modèle
        if model is not None:
            # Vérifier si c'est un dictionnaire complet ou juste les poids
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            else:
                model.load_state_dict(checkpoint, strict=strict)
        
        # Charger l'état de l'optimiseur
        if optimizer is not None and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    @staticmethod
    def get_best_checkpoint(checkpoint_dir: str, model_name: str = "model") -> Optional[str]:
        """
        Trouve le meilleur point de contrôle dans un répertoire.
        
        Args:
            checkpoint_dir: Répertoire contenant les points de contrôle.
            model_name: Nom du modèle.
            
        Returns:
            Chemin vers le meilleur point de contrôle, ou None si aucun trouvé.
        """
        best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
        
        if os.path.exists(best_path):
            return best_path
        
        # Si le meilleur modèle n'existe pas, chercher le dernier
        checkpoints = [f for f in os.listdir(checkpoint_dir) 
                      if f.startswith(model_name) and f.endswith('.pt')]
        
        if not checkpoints:
            return None
        
        # Trier par date de modification (le plus récent en premier)
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
        
        return os.path.join(checkpoint_dir, checkpoints[0])
    
    @staticmethod
    def list_checkpoints(checkpoint_dir: str, model_name: str = "model") -> List[Dict[str, Any]]:
        """
        Liste tous les points de contrôle dans un répertoire.
        
        Args:
            checkpoint_dir: Répertoire contenant les points de contrôle.
            model_name: Nom du modèle.
            
        Returns:
            Liste des informations sur les points de contrôle.
        """
        # Chercher tous les fichiers de point de contrôle
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                          if f.startswith(model_name) and f.endswith('.pt')]
        
        checkpoints = []
        for filename in checkpoint_files:
            filepath = os.path.join(checkpoint_dir, filename)
            stats = os.stat(filepath)
            
            # Extraire les informations de base
            info = {
                'filename': filename,
                'path': filepath,
                'size': stats.st_size,
                'modified': stats.st_mtime,
                'is_best': 'best' in filename
            }
            
            # Essayer de charger les métriques
            try:
                checkpoint = torch.load(filepath, map_location='cpu')
                if isinstance(checkpoint, dict):
                    if 'epoch' in checkpoint:
                        info['epoch'] = checkpoint['epoch']
                    if 'metrics' in checkpoint:
                        info['metrics'] = checkpoint['metrics']
            except Exception:
                pass
            
            checkpoints.append(info)
        
        # Trier par date de modification (le plus récent en premier)
        checkpoints.sort(key=lambda x: x['modified'], reverse=True)
        
        return checkpoints 