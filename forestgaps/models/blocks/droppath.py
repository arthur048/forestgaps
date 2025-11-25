"""
Implémentation du mécanisme DropPath pour la régularisation des réseaux profonds.

Ce module fournit une implémentation du mécanisme DropPath (Stochastic Depth)
pour améliorer la régularisation des réseaux résiduels profonds, tel que décrit
dans l'article "Deep Networks with Stochastic Depth" (Huang et al., 2016).
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)


class DropPath(nn.Module):
    """
    Mécanisme DropPath (Stochastic Depth) pour la régularisation des réseaux profonds.
    
    Cette technique abandonne aléatoirement des blocs entiers du réseau pendant
    l'entraînement avec une probabilité donnée, ce qui force les couches précédentes
    à apprendre des caractéristiques plus robustes.
    
    En mode évaluation, tous les chemins sont conservés.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        """
        Initialise le module DropPath.
        
        Args:
            drop_prob: Probabilité d'abandonner le bloc (entre 0 et 1)
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applique le mécanisme DropPath pendant l'entraînement.
        
        Args:
            x: Tensor d'entrée
            
        Returns:
            Tensor traité avec DropPath si en mode entraînement, 
            sinon le tensor original
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        # Créer un masque binaire pour décider quels échantillons du batch sont abandonnés
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # [B, 1, 1, 1]
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor < self.keep_prob
        
        # Mise à l'échelle de l'output pour conserver la même magnitude en moyenne
        output = x.div(self.keep_prob) * random_tensor
        return output


class DropPathScheduler:
    """
    Planificateur pour augmenter progressivement le taux de DropPath pendant l'entraînement.
    
    Cette classe permet d'augmenter graduellement le taux de DropPath au fil des époques,
    en utilisant différentes stratégies d'augmentation (linéaire, cosinus, etc.).
    Elle peut également appliquer des taux différents selon la profondeur des couches
    dans le réseau.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        start_prob: float = 0.0, 
        final_prob: float = 0.3, 
        epochs: int = 100, 
        strategy: str = 'linear',
        layer_wise: bool = True, 
        deeper_more_drop: bool = True
    ):
        """
        Initialise le planificateur DropPath.
        
        Args:
            model: Le modèle contenant les modules DropPath
            start_prob: Probabilité initiale de drop
            final_prob: Probabilité finale de drop (atteinte à la dernière époque)
            epochs: Nombre total d'époques
            strategy: Stratégie d'augmentation ('linear', 'cosine', 'step')
            layer_wise: Si True, applique des taux différents selon la profondeur
            deeper_more_drop: Si True, les couches plus profondes ont des taux plus élevés
        """
        self.model = model
        self.start_prob = start_prob
        self.final_prob = final_prob
        self.epochs = epochs
        self.strategy = strategy
        self.layer_wise = layer_wise
        self.deeper_more_drop = deeper_more_drop
        
        # Trouver tous les modules DropPath dans le modèle
        self.droppath_modules = {}
        self._find_droppath_modules(model)
        
        if not self.droppath_modules:
            logger.warning("Aucun module DropPath trouvé dans le modèle.")
        else:
            logger.info(f"Trouvé {len(self.droppath_modules)} modules DropPath dans le modèle.")
    
    def _find_droppath_modules(self, module: nn.Module, path: str = ""):
        """
        Recherche récursivement tous les modules DropPath dans le modèle.
        
        Args:
            module: Module à analyser
            path: Chemin courant dans la hiérarchie des modules
        """
        for name, child in module.named_children():
            current_path = f"{path}.{name}" if path else name
            if isinstance(child, DropPath):
                self.droppath_modules[current_path] = child
            else:
                self._find_droppath_modules(child, current_path)
    
    def step(self, epoch: int):
        """
        Met à jour les taux de DropPath en fonction de l'époque actuelle.
        
        Args:
            epoch: Époque courante (0-indexed)
        """
        if not self.droppath_modules:
            return
        
        progress = min(1.0, epoch / self.epochs)
        
        # Calculer le taux global selon la stratégie choisie
        if self.strategy == 'linear':
            global_rate = self.start_prob + progress * (self.final_prob - self.start_prob)
        elif self.strategy == 'cosine':
            global_rate = self.start_prob + 0.5 * (self.final_prob - self.start_prob) * (1 - torch.cos(torch.tensor(progress * torch.pi)).item())
        elif self.strategy == 'step':
            # Stratégie par paliers (25%, 50%, 75% des époques)
            if progress < 0.25:
                global_rate = self.start_prob
            elif progress < 0.5:
                global_rate = self.start_prob + 0.33 * (self.final_prob - self.start_prob)
            elif progress < 0.75:
                global_rate = self.start_prob + 0.66 * (self.final_prob - self.start_prob)
            else:
                global_rate = self.final_prob
        else:
            global_rate = self.final_prob  # Stratégie par défaut
        
        # Appliquer le taux aux modules DropPath
        if self.layer_wise and len(self.droppath_modules) > 1:
            # Si layer_wise est activé, on utilise des taux différents selon la profondeur
            sorted_paths = sorted(self.droppath_modules.keys())
            num_layers = len(sorted_paths)
            
            for i, path in enumerate(sorted_paths):
                # Calculer un facteur basé sur la position dans le réseau
                if self.deeper_more_drop:
                    # Les couches plus profondes ont des taux plus élevés
                    layer_factor = (i + 1) / num_layers
                else:
                    # Les couches moins profondes ont des taux plus élevés
                    layer_factor = 1.0 - (i / num_layers)
                
                # Appliquer le taux spécifique à la couche
                layer_rate = global_rate * layer_factor
                self.droppath_modules[path].drop_prob = layer_rate
                self.droppath_modules[path].keep_prob = 1.0 - layer_rate
        else:
            # Appliquer le même taux à tous les modules
            for module in self.droppath_modules.values():
                module.drop_prob = global_rate
                module.keep_prob = 1.0 - global_rate
        
        logger.debug(f"DropPath: époque {epoch}, taux global {global_rate:.4f}")


def train_with_droppath_scheduling(
    model: nn.Module, 
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epochs: int = 100,
    device: torch.device = None,
    start_prob: float = 0.0,
    final_prob: float = 0.3,
    strategy: str = 'linear',
    layer_wise: bool = True,
    callbacks: List[Callable] = None
):
    """
    Entraîne un modèle avec planification progressive du taux de DropPath.
    
    Args:
        model: Le modèle à entraîner
        train_loader: DataLoader pour les données d'entraînement
        val_loader: DataLoader pour les données de validation
        criterion: Fonction de perte
        optimizer: Optimiseur
        scheduler: Scheduler de learning rate (optionnel)
        epochs: Nombre total d'époques
        device: Appareil sur lequel effectuer l'entraînement
        start_prob: Probabilité initiale de DropPath
        final_prob: Probabilité finale de DropPath
        strategy: Stratégie d'augmentation du taux
        layer_wise: Si True, applique des taux différents selon la profondeur
        callbacks: Liste de fonctions de rappel à appeler après chaque époque
        
    Returns:
        dict: Historique d'entraînement avec métriques
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialiser le planificateur DropPath
    droppath_scheduler = DropPathScheduler(
        model, 
        start_prob=start_prob,
        final_prob=final_prob,
        epochs=epochs,
        strategy=strategy,
        layer_wise=layer_wise
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    for epoch in range(epochs):
        # Mettre à jour les taux de DropPath
        droppath_scheduler.step(epoch)
        
        # Appeler la fonction d'entraînement standard pour une époque
        # Note: cette fonction doit être implémentée ailleurs et retourner les métriques
        train_metrics = None  # train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = None    # validate(model, val_loader, criterion, device)
        
        # Mettre à jour le scheduler de learning rate si fourni
        if scheduler is not None:
            scheduler.step()
            
        # Enregistrer les métriques
        history['train_loss'].append(train_metrics['loss'] if train_metrics else None)
        history['val_loss'].append(val_metrics['loss'] if val_metrics else None)
        history['train_metrics'].append(train_metrics)
        history['val_metrics'].append(val_metrics)
        
        # Appeler les callbacks si fournis
        if callbacks:
            for callback in callbacks:
                callback(epoch, model, optimizer, train_metrics, val_metrics)
    
    return history 