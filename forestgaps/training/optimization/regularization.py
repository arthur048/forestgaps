"""
Module de techniques de régularisation pour l'entraînement.

Ce module fournit des classes et des fonctions pour implémenter différentes
techniques de régularisation afin d'améliorer la généralisation des modèles.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """
    Implémentation du Drop Path (Stochastic Depth).
    
    Cette technique est utilisée dans les architectures modernes comme
    les ViT et les Transformers pour améliorer la régularisation. Elle
    consiste à désactiver aléatoirement des blocs entiers pendant l'entraînement.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        """
        Initialise le module DropPath.
        
        Args:
            drop_prob: Probabilité de désactiver un chemin.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applique le DropPath lors de la passe avant.
        
        Args:
            x: Tensor d'entrée.
            
        Returns:
            Tensor après application du DropPath.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        # Garder le même tensor pour tous les éléments du batch
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Shape pour le broadcast
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarisation
        output = x.div(keep_prob) * random_tensor  # Mise à l'échelle
        
        return output


class DropPathScheduler:
    """
    Planificateur pour ajuster progressivement le taux de DropPath pendant l'entraînement.
    
    Cette classe permet d'augmenter progressivement le taux de DropPath
    au cours de l'entraînement, ce qui est bénéfique pour la stabilité et la convergence.
    """
    
    def __init__(self, model: nn.Module, start_prob: float = 0.0, final_prob: float = 0.3,
                 epochs: int = 100, strategy: str = 'linear', layer_wise: bool = True,
                 deeper_more_drop: bool = True):
        """
        Initialise le planificateur DropPath.
        
        Args:
            model: Modèle à régulariser.
            start_prob: Probabilité initiale de DropPath.
            final_prob: Probabilité finale de DropPath.
            epochs: Nombre total d'époques.
            strategy: Stratégie d'augmentation ('linear', 'cosine', 'exp').
            layer_wise: Appliquer des taux différents selon la couche.
            deeper_more_drop: Appliquer plus de dropout aux couches profondes.
        """
        self.model = model
        self.start_prob = start_prob
        self.final_prob = final_prob
        self.epochs = epochs
        self.strategy = strategy
        self.layer_wise = layer_wise
        self.deeper_more_drop = deeper_more_drop
        
        # Trouver tous les modules DropPath
        self.droppath_modules = []
        self._find_droppath_modules(model)
        
        # Initialiser les taux de base pour chaque module
        if layer_wise and self.droppath_modules:
            n_modules = len(self.droppath_modules)
            if deeper_more_drop:
                # Plus de dropout pour les couches profondes
                self.base_probs = [
                    final_prob * (i + 1) / n_modules for i in range(n_modules)
                ]
            else:
                # Même taux pour toutes les couches
                self.base_probs = [final_prob] * n_modules
        else:
            self.base_probs = [final_prob] * len(self.droppath_modules)
    
    def _find_droppath_modules(self, module: nn.Module, path: str = ""):
        """
        Trouve tous les modules DropPath dans le modèle.
        
        Args:
            module: Module à analyser.
            path: Chemin du module actuel.
        """
        for name, child in module.named_children():
            current_path = f"{path}.{name}" if path else name
            
            if isinstance(child, DropPath):
                self.droppath_modules.append((current_path, child))
            else:
                self._find_droppath_modules(child, current_path)
    
    def step(self, epoch: int) -> None:
        """
        Met à jour les taux de DropPath pour l'époque actuelle.
        
        Args:
            epoch: Époque actuelle.
        """
        if not self.droppath_modules:
            return
        
        # Calculer le facteur d'échelle selon la stratégie
        if self.strategy == 'linear':
            # Augmentation linéaire
            scale = min(1.0, epoch / (self.epochs * 0.8))
        elif self.strategy == 'cosine':
            # Augmentation en cosinus
            scale = 0.5 * (1 + np.cos(np.pi * (1 - min(1.0, epoch / (self.epochs * 0.8)))))
        elif self.strategy == 'exp':
            # Augmentation exponentielle
            scale = np.exp(np.log(1e-3) * (1 - min(1.0, epoch / (self.epochs * 0.8))))
        else:
            raise ValueError(f"Stratégie non reconnue: {self.strategy}")
        
        # Mettre à jour chaque module
        for i, (path, module) in enumerate(self.droppath_modules):
            target_prob = self.start_prob + (self.base_probs[i] - self.start_prob) * scale
            module.drop_prob = target_prob


class AdaptiveNorm(nn.Module):
    """
    Module de normalisation adaptative qui choisit automatiquement entre BatchNorm et GroupNorm.
    
    Ce module permet d'utiliser BatchNorm quand les batches sont suffisamment grands,
    et de basculer vers GroupNorm pour les petits batches.
    """
    
    def __init__(self, num_features: int, batch_size_threshold: int = 16, 
                 num_groups: int = 8, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        """
        Initialise le module de normalisation adaptative.
        
        Args:
            num_features: Nombre de features à normaliser.
            batch_size_threshold: Seuil pour basculer entre BatchNorm et GroupNorm.
            num_groups: Nombre de groupes pour GroupNorm.
            eps: Valeur epsilon pour la stabilité numérique.
            momentum: Momentum pour BatchNorm.
            affine: Appliquer des paramètres affines.
            track_running_stats: Suivre les statistiques pour BatchNorm.
        """
        super(AdaptiveNorm, self).__init__()
        
        # Créer les deux types de normalisation
        self.batch_norm = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum, 
            affine=affine, track_running_stats=track_running_stats
        )
        self.group_norm = nn.GroupNorm(
            num_groups, num_features, eps=eps, affine=affine
        )
        
        self.batch_size_threshold = batch_size_threshold
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applique la normalisation adaptative.
        
        Args:
            x: Tensor d'entrée.
            
        Returns:
            Tensor normalisé.
        """
        batch_size = x.size(0)
        
        if batch_size >= self.batch_size_threshold:
            # Utiliser BatchNorm pour les grands batches
            return self.batch_norm(x)
        else:
            # Utiliser GroupNorm pour les petits batches
            return self.group_norm(x)


class CompositeRegularization(nn.Module):
    """
    Module de régularisation composite combinant plusieurs techniques.
    
    Cette classe permet d'appliquer une combinaison optimale de techniques
    de régularisation, dont l'intensité peut être ajustée dynamiquement
    pendant l'entraînement.
    """
    
    def __init__(self, model: nn.Module, dropout_rate: float = 0.1,
                 weight_decay: float = 1e-4, drop_path_rate: float = 0.0,
                 label_smoothing: float = 0.0, stochastic_depth: bool = False,
                 spectral_norm: bool = False, auxilary_loss: bool = False):
        """
        Initialise le module de régularisation composite.
        
        Args:
            model: Modèle à régulariser.
            dropout_rate: Taux de dropout.
            weight_decay: Taux de weight decay.
            drop_path_rate: Taux de drop path.
            label_smoothing: Taux de label smoothing.
            stochastic_depth: Activer stochastic depth.
            spectral_norm: Activer spectral normalization.
            auxilary_loss: Activer les pertes auxiliaires.
        """
        super(CompositeRegularization, self).__init__()
        
        self.model = model
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.drop_path_rate = drop_path_rate
        self.label_smoothing = label_smoothing
        self.stochastic_depth = stochastic_depth
        self.spectral_norm = spectral_norm
        self.auxilary_loss = auxilary_loss
        
        # Appliquer les techniques de régularisation
        self._apply_dropout()
        self._apply_drop_path()
        if spectral_norm:
            self._apply_spectral_norm()
    
    def _apply_dropout(self) -> None:
        """
        Applique le dropout à tous les modules appropriés.
        """
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.dropout_rate
    
    def _apply_drop_path(self) -> None:
        """
        Applique le drop path aux modules appropriés.
        """
        if self.stochastic_depth:
            # Chercher les blocs résiduels ou similaires
            depth_modules = []
            for name, module in self.model.named_modules():
                if any(block_type in name.lower() for block_type in ['block', 'layer', 'residual']):
                    if hasattr(module, 'downsample') or hasattr(module, 'shortcut'):
                        depth_modules.append((name, module))
            
            # Appliquer DropPath avec taux progressif
            n_modules = len(depth_modules)
            for i, (name, module) in enumerate(depth_modules):
                drop_prob = self.drop_path_rate * (i + 1) / n_modules
                
                # Chercher l'endroit approprié pour insérer DropPath
                if hasattr(module, 'apply_droppath'):
                    module.apply_droppath = True
                    module.drop_path = DropPath(drop_prob)
    
    def _apply_spectral_norm(self) -> None:
        """
        Applique la normalisation spectrale aux couches de convolution.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Remplacer le module par sa version avec spectral norm
                parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
                parent = self.model if parent_name == '' else self._get_module(self.model, parent_name)
                
                if parent is not None:
                    setattr(parent, child_name, nn.utils.spectral_norm(module))
    
    def _get_module(self, model: nn.Module, path: str) -> Optional[nn.Module]:
        """
        Récupère un module à partir de son chemin.
        
        Args:
            model: Module parent.
            path: Chemin du module à récupérer.
            
        Returns:
            Module correspondant ou None si non trouvé.
        """
        parts = path.split('.')
        current = model
        
        for part in parts:
            if not hasattr(current, part):
                return None
            current = getattr(current, part)
        
        return current
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant du modèle avec régularisation.
        
        Args:
            x: Tensor d'entrée.
            
        Returns:
            Sortie du modèle.
        """
        return self.model(x)
    
    def get_parameters_with_weight_decay(self) -> List[Dict[str, List]]:
        """
        Sépare les paramètres qui doivent avoir du weight decay de ceux qui n'en ont pas.
        
        Returns:
            Liste de dictionnaires de paramètres pour l'optimiseur.
        """
        decay, no_decay = [], []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Les biais et les paramètres de normalisation ne doivent pas avoir de weight decay
            if param.ndim <= 1 or 'bn' in name or 'bias' in name or 'norm' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        
        return [
            {'params': decay, 'weight_decay': self.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]


class GradientClipping:
    """
    Classe pour appliquer le gradient clipping pendant l'entraînement.
    
    Cette technique limite la norme des gradients pour éviter les
    explosions de gradient et stabiliser l'entraînement.
    """
    
    def __init__(self, model: nn.Module, clip_value: float = 1.0, clip_norm: bool = True,
                 monitor: bool = False):
        """
        Initialise le gradient clipping.
        
        Args:
            model: Modèle à entraîner.
            clip_value: Valeur de clipping.
            clip_norm: Utiliser clip par norme (True) ou par valeur (False).
            monitor: Suivre les statistiques des gradients.
        """
        self.model = model
        self.clip_value = clip_value
        self.clip_norm = clip_norm
        self.monitor = monitor
        
        # Statistiques de monitoring
        self.grad_norms = []
        self.grad_max = []
        self.grad_min = []
        self.grad_mean = []
    
    def clip_gradients(self) -> float:
        """
        Applique le gradient clipping.
        
        Returns:
            Norme des gradients avant clipping.
        """
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        
        if not parameters:
            return 0.0
        
        # Calculer les statistiques si nécessaire
        if self.monitor:
            grad_max = max(p.grad.data.max().item() for p in parameters)
            grad_min = min(p.grad.data.min().item() for p in parameters)
            grad_mean = np.mean([p.grad.data.mean().item() for p in parameters])
            
            self.grad_max.append(grad_max)
            self.grad_min.append(grad_min)
            self.grad_mean.append(grad_mean)
        
        # Appliquer le clipping
        if self.clip_norm:
            # Clipping par norme
            total_norm = torch.nn.utils.clip_grad_norm_(
                parameters, self.clip_value
            )
            if self.monitor:
                self.grad_norms.append(total_norm.item())
            return total_norm.item()
        else:
            # Clipping par valeur
            torch.nn.utils.clip_grad_value_(parameters, self.clip_value)
            
            # Calculer la norme pour le suivi
            total_norm = 0.0
            for p in parameters:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if self.monitor:
                self.grad_norms.append(total_norm)
            
            return total_norm
    
    def get_stats(self) -> Dict[str, List[float]]:
        """
        Récupère les statistiques de gradient.
        
        Returns:
            Dictionnaire des statistiques.
        """
        if not self.monitor:
            return {}
        
        return {
            'grad_norms': self.grad_norms,
            'grad_max': self.grad_max,
            'grad_min': self.grad_min,
            'grad_mean': self.grad_mean
        }


def apply_adaptive_normalization(model: nn.Module, batch_size: int) -> None:
    """
    Remplace les couches de normalisation par des couches adaptatives.
    
    Args:
        model: Modèle à modifier.
        batch_size: Taille du batch pour déterminer le seuil d'adaptation.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Remplacer par AdaptiveNorm
            adaptive_norm = AdaptiveNorm(
                num_features=module.num_features,
                batch_size_threshold=max(16, batch_size // 4),
                num_groups=min(32, module.num_features // 4),
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats
            )
            
            # Copier les paramètres si nécessaire
            if module.affine:
                adaptive_norm.batch_norm.weight.data = module.weight.data.clone()
                adaptive_norm.batch_norm.bias.data = module.bias.data.clone()
                adaptive_norm.group_norm.weight.data = module.weight.data.clone()
                adaptive_norm.group_norm.bias.data = module.bias.data.clone()
            
            # Remplacer le module
            setattr(model, name, adaptive_norm)
        else:
            # Récursion pour les sous-modules
            apply_adaptive_normalization(module, batch_size) 