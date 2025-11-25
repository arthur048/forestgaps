"""
Module de base pour les callbacks d'entraînement.

Ce module fournit la classe de base pour tous les types de callbacks
qui peuvent être utilisés pendant l'entraînement des modèles.
"""

from typing import Dict, Any, Optional, List, Union


class Callback:
    """
    Classe de base pour les callbacks d'entraînement.
    
    Cette classe définit l'interface que tous les callbacks doivent implémenter.
    Les callbacks permettent d'exécuter du code à différents moments de l'entraînement,
    comme le début/fin de l'entraînement, le début/fin d'une époque, etc.
    """
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé au début de l'entraînement.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        pass
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de l'entraînement.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé au début de chaque époque.
        
        Args:
            epoch: Numéro de l'époque actuelle.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de chaque époque.
        
        Args:
            epoch: Numéro de l'époque actuelle.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé au début de chaque batch.
        
        Args:
            batch: Numéro du batch actuel.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de chaque batch.
        
        Args:
            batch: Numéro du batch actuel.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        pass
    
    def on_validation_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé au début de la validation.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        pass
    
    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de la validation.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        pass
    
    def on_test_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé au début du test.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état du test.
        """
        pass
    
    def on_test_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin du test.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état du test.
        """
        pass
    
    def on_train_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé au début de chaque batch d'entraînement.
        
        Args:
            batch: Numéro du batch actuel.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        self.on_batch_begin(batch, logs)
    
    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de chaque batch d'entraînement.
        
        Args:
            batch: Numéro du batch actuel.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        self.on_batch_end(batch, logs)
    
    def on_validation_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé au début de chaque batch de validation.
        
        Args:
            batch: Numéro du batch actuel.
            logs: Dictionnaire contenant des informations sur l'état de la validation.
        """
        pass
    
    def on_validation_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de chaque batch de validation.
        
        Args:
            batch: Numéro du batch actuel.
            logs: Dictionnaire contenant des informations sur l'état de la validation.
        """
        pass
    
    def on_test_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé au début de chaque batch de test.
        
        Args:
            batch: Numéro du batch actuel.
            logs: Dictionnaire contenant des informations sur l'état du test.
        """
        pass
    
    def on_test_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de chaque batch de test.
        
        Args:
            batch: Numéro du batch actuel.
            logs: Dictionnaire contenant des informations sur l'état du test.
        """
        pass


class CallbackList:
    """
    Gestionnaire de liste de callbacks.
    
    Cette classe permet de regrouper plusieurs callbacks et de les appeler
    séquentiellement lors des différents événements de l'entraînement.
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Initialise la liste de callbacks.
        
        Args:
            callbacks: Liste des callbacks à gérer.
        """
        self.callbacks = callbacks if callbacks is not None else []
    
    def append(self, callback: Callback) -> None:
        """
        Ajoute un callback à la liste.
        
        Args:
            callback: Callback à ajouter.
        """
        self.callbacks.append(callback)
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelle la méthode on_train_begin de tous les callbacks.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelle la méthode on_train_end de tous les callbacks.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelle la méthode on_epoch_begin de tous les callbacks.
        
        Args:
            epoch: Numéro de l'époque actuelle.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelle la méthode on_epoch_end de tous les callbacks.
        
        Args:
            epoch: Numéro de l'époque actuelle.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelle la méthode on_batch_begin de tous les callbacks.
        
        Args:
            batch: Numéro du batch actuel.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelle la méthode on_batch_end de tous les callbacks.
        
        Args:
            batch: Numéro du batch actuel.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
    
    def on_validation_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelle la méthode on_validation_begin de tous les callbacks.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_validation_begin(logs)
    
    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelle la méthode on_validation_end de tous les callbacks.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_validation_end(logs)
    
    def __iter__(self):
        """
        Itérateur sur les callbacks.
        
        Returns:
            Itérateur sur les callbacks.
        """
        return iter(self.callbacks) 