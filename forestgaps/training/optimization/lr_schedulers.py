"""
Module de schedulers de taux d'apprentissage pour l'optimisation.

Ce module fournit des fonctions et des classes pour créer et gérer différents
schedulers de taux d'apprentissage pour l'entraînement des modèles.
"""

import math
from typing import Dict, Any, Optional, List, Union, Callable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Scheduler qui combine un warm-up linéaire avec un cosine annealing.
    
    Ce scheduler est particulièrement efficace pour l'entraînement des
    modèles de segmentation, en commençant par augmenter linéairement le
    taux d'apprentissage puis en le diminuant selon un cosinus.
    """
    
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, max_epochs: int,
                 warmup_start_lr: float = 1e-8, eta_min: float = 1e-8,
                 last_epoch: int = -1):
        """
        Initialise le scheduler.
        
        Args:
            optimizer: Optimiseur à utiliser.
            warmup_epochs: Nombre d'époques pour le warm-up.
            max_epochs: Nombre total d'époques.
            warmup_start_lr: Taux d'apprentissage initial pour le warm-up.
            eta_min: Taux d'apprentissage minimum.
            last_epoch: Dernière époque.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """
        Calcule le taux d'apprentissage à l'époque actuelle.
        
        Returns:
            Liste des taux d'apprentissage pour chaque groupe de paramètres.
        """
        if self.last_epoch < self.warmup_epochs:
            # Warm-up linéaire
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * 
                   self.last_epoch / self.warmup_epochs
                   for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / 
                               (self.max_epochs - self.warmup_epochs))) / 2
                   for base_lr in self.base_lrs]


class CyclicCosineAnnealingLR(_LRScheduler):
    """
    Scheduler qui applique un cosine annealing cyclique avec redémarrage.
    
    Ce scheduler diminue le taux d'apprentissage selon un cosinus, puis le
    réinitialise à la fin de chaque cycle, permettant au modèle de sortir
    des minima locaux.
    """
    
    def __init__(self, optimizer: Optimizer, cycle_epochs: int, cycles: int = 1,
                 cycle_mult: float = 1.0, eta_min: float = 1e-8, last_epoch: int = -1):
        """
        Initialise le scheduler.
        
        Args:
            optimizer: Optimiseur à utiliser.
            cycle_epochs: Nombre d'époques par cycle.
            cycles: Nombre de cycles à effectuer.
            cycle_mult: Facteur multiplicatif pour la longueur des cycles successifs.
            eta_min: Taux d'apprentissage minimum.
            last_epoch: Dernière époque.
        """
        self.cycle_epochs = cycle_epochs
        self.cycles = cycles
        self.cycle_mult = cycle_mult
        self.eta_min = eta_min
        
        # Calculer la longueur totale des cycles
        self.total_epochs = 0
        cycle_length = cycle_epochs
        for i in range(cycles):
            self.total_epochs += cycle_length
            cycle_length = int(cycle_length * cycle_mult)
        
        super(CyclicCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """
        Calcule le taux d'apprentissage à l'époque actuelle.
        
        Returns:
            Liste des taux d'apprentissage pour chaque groupe de paramètres.
        """
        # Déterminer le cycle actuel et l'époque dans ce cycle
        epoch = self.last_epoch
        current_cycle = 0
        cycle_length = self.cycle_epochs
        cycle_start = 0
        
        while epoch >= cycle_start + cycle_length and current_cycle < self.cycles - 1:
            cycle_start += cycle_length
            current_cycle += 1
            cycle_length = int(cycle_length * self.cycle_mult)
        
        # Calculer la position relative dans le cycle
        cycle_epoch = epoch - cycle_start
        relative_pos = cycle_epoch / cycle_length
        
        # Appliquer le cosine annealing
        return [self.eta_min + (base_lr - self.eta_min) * 
               (1 + math.cos(math.pi * relative_pos)) / 2
               for base_lr in self.base_lrs]


class LinearWarmupReduceLROnPlateau:
    """
    Scheduler qui combine un warm-up linéaire avec une réduction sur plateau.
    
    Ce scheduler est utile pour les problèmes où la performance peut stagner,
    en diminuant le taux d'apprentissage lorsqu'une métrique ne s'améliore plus.
    """
    
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, patience: int = 5,
                 factor: float = 0.1, min_lr: float = 1e-8, mode: str = 'min',
                 threshold: float = 1e-4, cooldown: int = 0, verbose: bool = False):
        """
        Initialise le scheduler.
        
        Args:
            optimizer: Optimiseur à utiliser.
            warmup_epochs: Nombre d'époques pour le warm-up.
            patience: Nombre d'époques à attendre avant de réduire le LR.
            factor: Facteur de réduction du LR.
            min_lr: LR minimum.
            mode: Mode d'évaluation ('min' ou 'max').
            threshold: Seuil pour considérer une amélioration.
            cooldown: Nombre d'époques à attendre après une réduction.
            verbose: Afficher des informations lors des réductions.
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.mode = mode
        self.threshold = threshold
        self.cooldown = cooldown
        self.verbose = verbose
        
        # Paramètres internes
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.warmup_start_lr = min_lr
        self.last_epoch = -1
        self.cooldown_counter = 0
        self.wait = 0
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.plateau_scheduler = None
    
    def step(self, metrics=None):
        """
        Met à jour le taux d'apprentissage.
        
        Args:
            metrics: Métrique utilisée pour la réduction sur plateau.
        """
        self.last_epoch += 1
        
        if self.last_epoch < self.warmup_epochs:
            # Warm-up linéaire
            lr_scale = self.last_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.warmup_start_lr + (self.base_lrs[i] - self.warmup_start_lr) * lr_scale
        else:
            # Initialiser le plateau scheduler si nécessaire
            if self.plateau_scheduler is None:
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                self.plateau_scheduler = ReduceLROnPlateau(
                    self.optimizer, mode=self.mode, factor=self.factor, 
                    patience=self.patience, threshold=self.threshold,
                    threshold_mode='rel', cooldown=self.cooldown,
                    min_lr=self.min_lr, verbose=self.verbose
                )
            
            # Mise à jour par le plateau scheduler
            self.plateau_scheduler.step(metrics)
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Renvoie l'état du scheduler.
        
        Returns:
            Dictionnaire contenant l'état du scheduler.
        """
        state = {
            'warmup_epochs': self.warmup_epochs,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch,
            'warmup_start_lr': self.warmup_start_lr,
            'patience': self.patience,
            'factor': self.factor,
            'min_lr': self.min_lr,
            'mode': self.mode,
            'threshold': self.threshold,
            'cooldown': self.cooldown,
            'best': self.best,
            'wait': self.wait,
            'cooldown_counter': self.cooldown_counter
        }
        
        # Ajouter l'état du plateau scheduler si existant
        if self.plateau_scheduler is not None:
            state['plateau_scheduler'] = self.plateau_scheduler.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Charge l'état du scheduler.
        
        Args:
            state_dict: Dictionnaire contenant l'état du scheduler.
        """
        self.warmup_epochs = state_dict['warmup_epochs']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        self.patience = state_dict['patience']
        self.factor = state_dict['factor']
        self.min_lr = state_dict['min_lr']
        self.mode = state_dict['mode']
        self.threshold = state_dict['threshold']
        self.cooldown = state_dict['cooldown']
        self.best = state_dict['best']
        self.wait = state_dict['wait']
        self.cooldown_counter = state_dict['cooldown_counter']
        
        # Charger l'état du plateau scheduler si existant
        if self.last_epoch >= self.warmup_epochs and 'plateau_scheduler' in state_dict:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.plateau_scheduler = ReduceLROnPlateau(
                self.optimizer, mode=self.mode, factor=self.factor, 
                patience=self.patience, threshold=self.threshold,
                threshold_mode='rel', cooldown=self.cooldown,
                min_lr=self.min_lr, verbose=self.verbose
            )
            self.plateau_scheduler.load_state_dict(state_dict['plateau_scheduler'])


def create_scheduler(optimizer: Optimizer, config: Any, 
                    train_loader_len: Optional[int] = None) -> Union[_LRScheduler, LinearWarmupReduceLROnPlateau]:
    """
    Crée un scheduler de taux d'apprentissage à partir de la configuration.
    
    Args:
        optimizer: Optimiseur à utiliser.
        config: Configuration contenant les paramètres du scheduler.
        train_loader_len: Longueur du DataLoader d'entraînement (pour OneCycle).
        
    Returns:
        Scheduler configuré.
    """
    scheduler_type = getattr(config, 'scheduler_type', 'cosine')
    
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=getattr(config, 'epochs', 100),
            eta_min=getattr(config, 'min_lr', 1e-6)
        )
    
    elif scheduler_type == 'cosine_warm':
        return LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=getattr(config, 'warmup_epochs', 5),
            max_epochs=getattr(config, 'epochs', 100),
            warmup_start_lr=getattr(config, 'warmup_start_lr', 1e-6),
            eta_min=getattr(config, 'min_lr', 1e-6)
        )
    
    elif scheduler_type == 'cosine_restart':
        return CyclicCosineAnnealingLR(
            optimizer,
            cycle_epochs=getattr(config, 'cycle_epochs', 30),
            cycles=getattr(config, 'cycles', 3),
            cycle_mult=getattr(config, 'cycle_mult', 1.0),
            eta_min=getattr(config, 'min_lr', 1e-6)
        )
    
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=getattr(config, 'mode', 'min'),
            factor=getattr(config, 'factor', 0.1),
            patience=getattr(config, 'patience', 10),
            threshold=getattr(config, 'threshold', 1e-4),
            threshold_mode=getattr(config, 'threshold_mode', 'rel'),
            cooldown=getattr(config, 'cooldown', 0),
            min_lr=getattr(config, 'min_lr', 1e-6),
            verbose=getattr(config, 'verbose', True)
        )
    
    elif scheduler_type == 'warmup_plateau':
        return LinearWarmupReduceLROnPlateau(
            optimizer,
            warmup_epochs=getattr(config, 'warmup_epochs', 5),
            patience=getattr(config, 'patience', 10),
            factor=getattr(config, 'factor', 0.1),
            min_lr=getattr(config, 'min_lr', 1e-6),
            mode=getattr(config, 'mode', 'min'),
            threshold=getattr(config, 'threshold', 1e-4),
            cooldown=getattr(config, 'cooldown', 0),
            verbose=getattr(config, 'verbose', True)
        )
    
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(config, 'step_size', 30),
            gamma=getattr(config, 'gamma', 0.1)
        )
    
    elif scheduler_type == 'onecycle':
        if train_loader_len is None:
            raise ValueError("train_loader_len doit être fourni pour le scheduler OneCycleLR")
        
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=getattr(config, 'max_lr', optimizer.param_groups[0]['lr'] * 10),
            total_steps=getattr(config, 'epochs', 100) * train_loader_len,
            pct_start=getattr(config, 'pct_start', 0.3),
            anneal_strategy=getattr(config, 'anneal_strategy', 'cos'),
            div_factor=getattr(config, 'div_factor', 25.0),
            final_div_factor=getattr(config, 'final_div_factor', 1e4)
        )
    
    else:
        raise ValueError(f"Type de scheduler non reconnu: {scheduler_type}")


class WarmupScheduler:
    """
    Un wrapper pour ajouter un warmup à n'importe quel scheduler existant.
    
    Cette classe permet d'ajouter une phase de warmup à n'importe quel scheduler
    de taux d'apprentissage.
    """
    
    def __init__(self, optimizer: Optimizer, scheduler: _LRScheduler,
                 warmup_epochs: int, warmup_start_lr: float = 1e-8):
        """
        Initialise le WarmupScheduler.
        
        Args:
            optimizer: Optimiseur à utiliser.
            scheduler: Scheduler de base à wraper.
            warmup_epochs: Nombre d'époques pour le warm-up.
            warmup_start_lr: Taux d'apprentissage initial pour le warm-up.
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        
        # Paramètres internes
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = -1
        
        # Initialiser les taux d'apprentissage
        self.step()
    
    def step(self, metrics=None):
        """
        Met à jour le taux d'apprentissage.
        
        Args:
            metrics: Métrique utilisée par le scheduler de base.
        """
        self.last_epoch += 1
        
        if self.last_epoch < self.warmup_epochs:
            # Warm-up linéaire
            lr_scale = self.last_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.warmup_start_lr + (self.base_lrs[i] - self.warmup_start_lr) * lr_scale
        else:
            # Utiliser le scheduler de base
            if hasattr(self.scheduler, 'step'):
                if 'metrics' in self.scheduler.step.__code__.co_varnames:
                    self.scheduler.step(metrics)
                else:
                    self.scheduler.step()
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Renvoie l'état du scheduler.
        
        Returns:
            Dictionnaire contenant l'état du scheduler.
        """
        state = {
            'warmup_epochs': self.warmup_epochs,
            'warmup_start_lr': self.warmup_start_lr,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch
        }
        
        # Ajouter l'état du scheduler de base
        if hasattr(self.scheduler, 'state_dict'):
            state['scheduler_state'] = self.scheduler.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Charge l'état du scheduler.
        
        Args:
            state_dict: Dictionnaire contenant l'état du scheduler.
        """
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']
        
        # Charger l'état du scheduler de base
        if 'scheduler_state' in state_dict and hasattr(self.scheduler, 'load_state_dict'):
            self.scheduler.load_state_dict(state_dict['scheduler_state']) 