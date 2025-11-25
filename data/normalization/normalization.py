"""
Module de normalisation des données pour la détection des trouées forestières.

Ce module fournit des classes et fonctions pour normaliser les données et les
exporter avec les modèles pour une utilisation en production.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.normalization.statistics import NormalizationStatistics
from data.normalization.strategies import (
    NormalizationStrategy, NormalizationMethod,
    MinMaxNormalization, ZScoreNormalization, RobustNormalization,
    AdaptiveNormalization, BatchNormStrategy,
    create_normalization_strategy
)

# Configuration du logger
logger = logging.getLogger(__name__)


class NormalizationLayer(nn.Module):
    """
    Couche de normalisation intégrable dans un modèle PyTorch.
    
    Cette couche encapsule une stratégie de normalisation pour qu'elle
    puisse être utilisée dans un modèle et exportée avec lui.
    """
    
    def __init__(
        self,
        strategy: NormalizationStrategy,
        trainable: bool = False,
        epsilon: float = 1e-8
    ):
        """
        Initialise la couche de normalisation.
        
        Args:
            strategy: Stratégie de normalisation à encapsuler
            trainable: Si True, les paramètres de normalisation sont apprenables
            epsilon: Valeur minimale pour éviter la division par zéro
        """
        super().__init__()
        self.strategy = strategy
        self.trainable = trainable
        self.epsilon = epsilon
        
        # Récupère les paramètres de la stratégie
        params = strategy.get_params()
        self.method = params.get('method', NormalizationMethod.MINMAX)
        
        # Crée les paramètres PyTorch selon la méthode de normalisation
        if self.method == NormalizationMethod.MINMAX:
            self.register_buffer('min_val', torch.tensor(params['min_val'], dtype=torch.float32))
            self.register_buffer('max_val', torch.tensor(params['max_val'], dtype=torch.float32))
            self.register_buffer('target_min', torch.tensor(params.get('target_min', 0.0), dtype=torch.float32))
            self.register_buffer('target_max', torch.tensor(params.get('target_max', 1.0), dtype=torch.float32))
            
            if trainable:
                self.scale = nn.Parameter(torch.tensor(params['scale'], dtype=torch.float32))
                self.shift = nn.Parameter(torch.tensor(params['shift'], dtype=torch.float32))
            else:
                self.register_buffer('scale', torch.tensor(params['scale'], dtype=torch.float32))
                self.register_buffer('shift', torch.tensor(params['shift'], dtype=torch.float32))
                
        elif self.method == NormalizationMethod.ZSCORE:
            self.register_buffer('mean', torch.tensor(params['mean'], dtype=torch.float32))
            self.register_buffer('std', torch.tensor(params['std'], dtype=torch.float32))
            
            if trainable:
                self.scale = nn.Parameter(torch.tensor(1.0 / (params['std'] + epsilon), dtype=torch.float32))
                self.shift = nn.Parameter(torch.tensor(-params['mean'] / (params['std'] + epsilon), dtype=torch.float32))
            else:
                self.register_buffer('scale', torch.tensor(1.0 / (params['std'] + epsilon), dtype=torch.float32))
                self.register_buffer('shift', torch.tensor(-params['mean'] / (params['std'] + epsilon), dtype=torch.float32))
                
        elif self.method == NormalizationMethod.ROBUST:
            self.register_buffer('median', torch.tensor(params['median'], dtype=torch.float32))
            self.register_buffer('p1', torch.tensor(params['p1'], dtype=torch.float32))
            self.register_buffer('p99', torch.tensor(params['p99'], dtype=torch.float32))
            self.register_buffer('center', torch.tensor(params['center'], dtype=torch.float32))
            
            if trainable:
                self.scale_low = nn.Parameter(torch.tensor(params['scale_low'], dtype=torch.float32))
                self.scale_high = nn.Parameter(torch.tensor(params['scale_high'], dtype=torch.float32))
            else:
                self.register_buffer('scale_low', torch.tensor(params['scale_low'], dtype=torch.float32))
                self.register_buffer('scale_high', torch.tensor(params['scale_high'], dtype=torch.float32))
        
        # Pour les autres méthodes, on délègue à la stratégie
        else:
            self.encapsulated_strategy = True
            # Si c'est un BatchNormStrategy, récupère le module BatchNorm
            if hasattr(strategy, 'batch_norm'):
                self.batch_norm = strategy.batch_norm
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalise les données en utilisant la stratégie encapsulée.
        
        Args:
            x: Tensor d'entrée à normaliser
            
        Returns:
            Tensor normalisé
        """
        # Pour les méthodes standard, implémente la normalisation directement
        if self.method == NormalizationMethod.MINMAX:
            # Écrête les valeurs si nécessaire
            if hasattr(self.strategy, 'clip_values') and self.strategy.clip_values:
                x = torch.clamp(x, self.min_val, self.max_val)
            
            # Applique la normalisation
            return x * self.scale + self.shift
            
        elif self.method == NormalizationMethod.ZSCORE:
            # Applique la normalisation z-score
            normalized = x * self.scale + self.shift
            
            # Écrête les valeurs si nécessaire
            if hasattr(self.strategy, 'clip_sigma') and self.strategy.clip_sigma is not None:
                normalized = torch.clamp(normalized, -self.strategy.clip_sigma, self.strategy.clip_sigma)
                
            return normalized
            
        elif self.method == NormalizationMethod.ROBUST:
            # Écrête les valeurs si nécessaire
            if hasattr(self.strategy, 'clip_values') and self.strategy.clip_values:
                x = torch.clamp(x, self.p1, self.p99)
            
            # Crée des masques pour les valeurs inférieures et supérieures à la médiane
            mask_low = x < self.median
            mask_high = ~mask_low
            
            # Applique la normalisation par parties
            result = torch.zeros_like(x)
            result[mask_low] = (x[mask_low] - self.median) * self.scale_low + self.center
            result[mask_high] = (x[mask_high] - self.median) * self.scale_high + self.center
            
            return result
            
        # Pour les autres méthodes, délègue à la stratégie
        else:
            # Si c'est un BatchNormStrategy, utilise directement le module BatchNorm
            if hasattr(self, 'batch_norm'):
                # Assure que le tensor a 4 dimensions [B, C, H, W]
                input_shape = x.shape
                if x.dim() == 3:  # [B, H, W]
                    x = x.unsqueeze(1)  # [B, 1, H, W]
                elif x.dim() == 2:  # [H, W]
                    x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                # Applique BatchNorm
                normalized = self.batch_norm(x)
                
                # Restaure la forme d'origine
                if len(input_shape) < normalized.dim():
                    normalized = normalized.reshape(input_shape)
                
                return normalized
            else:
                return self.strategy.normalize(x)
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dénormalise les données en utilisant la stratégie inverse.
        
        Args:
            x: Tensor d'entrée à dénormaliser
            
        Returns:
            Tensor dénormalisé
        """
        if self.method == NormalizationMethod.MINMAX:
            # Applique la dénormalisation
            denormalized = (x - self.shift) / self.scale
            
            # Écrête les valeurs si nécessaire
            if hasattr(self.strategy, 'clip_values') and self.strategy.clip_values:
                denormalized = torch.clamp(denormalized, self.min_val, self.max_val)
                
            return denormalized
            
        elif self.method == NormalizationMethod.ZSCORE:
            # Dénormalisation z-score
            return x * self.std + self.mean
            
        elif self.method == NormalizationMethod.ROBUST:
            # Crée des masques pour les valeurs inférieures et supérieures au centre
            mask_low = x < self.center
            mask_high = ~mask_low
            
            # Applique la dénormalisation par parties
            result = torch.zeros_like(x)
            result[mask_low] = (x[mask_low] - self.center) / self.scale_low + self.median
            result[mask_high] = (x[mask_high] - self.center) / self.scale_high + self.median
            
            # Écrête les valeurs si nécessaire
            if hasattr(self.strategy, 'clip_values') and self.strategy.clip_values:
                result = torch.clamp(result, self.p1, self.p99)
                
            return result
            
        # Pour les autres méthodes, délègue à la stratégie
        else:
            return self.strategy.denormalize(x)
    
    def get_strategy(self) -> NormalizationStrategy:
        """
        Retourne la stratégie de normalisation encapsulée.
        
        Returns:
            Stratégie de normalisation
        """
        return self.strategy
    
    def export_params(self) -> Dict[str, Any]:
        """
        Exporte les paramètres de normalisation pour une utilisation externe.
        
        Returns:
            Dictionnaire contenant les paramètres exportables
        """
        params = {'method': self.method}
        
        if self.method == NormalizationMethod.MINMAX:
            params.update({
                'min_val': self.min_val.item(),
                'max_val': self.max_val.item(),
                'target_min': self.target_min.item(),
                'target_max': self.target_max.item(),
                'scale': self.scale.item(),
                'shift': self.shift.item(),
                'clip_values': getattr(self.strategy, 'clip_values', True)
            })
        elif self.method == NormalizationMethod.ZSCORE:
            params.update({
                'mean': self.mean.item(),
                'std': self.std.item(),
                'scale': self.scale.item(),
                'shift': self.shift.item(),
                'clip_sigma': getattr(self.strategy, 'clip_sigma', None)
            })
        elif self.method == NormalizationMethod.ROBUST:
            params.update({
                'median': self.median.item(),
                'p1': self.p1.item(),
                'p99': self.p99.item(),
                'center': self.center.item(),
                'scale_low': self.scale_low.item(),
                'scale_high': self.scale_high.item(),
                'clip_values': getattr(self.strategy, 'clip_values', True)
            })
        elif hasattr(self, 'batch_norm'):
            params.update({
                'num_features': self.batch_norm.num_features,
                'eps': self.batch_norm.eps,
                'momentum': self.batch_norm.momentum,
                'affine': self.batch_norm.affine,
                'track_running_stats': self.batch_norm.track_running_stats
            })
            
            if self.batch_norm.track_running_stats:
                params.update({
                    'running_mean': self.batch_norm.running_mean.detach().cpu().numpy().tolist(),
                    'running_var': self.batch_norm.running_var.detach().cpu().numpy().tolist()
                })
            
            if self.batch_norm.affine:
                params.update({
                    'weight': self.batch_norm.weight.detach().cpu().numpy().tolist(),
                    'bias': self.batch_norm.bias.detach().cpu().numpy().tolist()
                })
        else:
            # Pour les autres méthodes, récupère les paramètres de la stratégie
            params.update(self.strategy.get_params())
        
        return params
    
    def save(self, file_path: str) -> None:
        """
        Sauvegarde les paramètres de normalisation dans un fichier JSON.
        
        Args:
            file_path: Chemin où sauvegarder le fichier JSON
        """
        params = self.export_params()
        
        try:
            # Crée le répertoire si nécessaire
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Sauvegarde les paramètres au format JSON
            with open(file_path, 'w') as f:
                json.dump(params, f, indent=2)
                
            logger.info(f"Paramètres de normalisation sauvegardés dans {file_path}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des paramètres de normalisation: {str(e)}")
    
    @classmethod
    def from_strategy(cls, strategy: NormalizationStrategy, trainable: bool = False) -> 'NormalizationLayer':
        """
        Crée une couche de normalisation à partir d'une stratégie.
        
        Args:
            strategy: Stratégie de normalisation
            trainable: Si True, les paramètres de normalisation sont apprenables
            
        Returns:
            Instance de NormalizationLayer
        """
        return cls(strategy=strategy, trainable=trainable)
    
    @classmethod
    def from_stats(
        cls,
        stats: NormalizationStatistics,
        method: Union[str, NormalizationMethod] = 'adaptive',
        trainable: bool = False,
        **kwargs
    ) -> 'NormalizationLayer':
        """
        Crée une couche de normalisation à partir de statistiques.
        
        Args:
            stats: Statistiques de normalisation
            method: Méthode de normalisation à utiliser
            trainable: Si True, les paramètres de normalisation sont apprenables
            **kwargs: Arguments supplémentaires pour la stratégie
            
        Returns:
            Instance de NormalizationLayer
        """
        strategy = create_normalization_strategy(method=method, stats=stats, **kwargs)
        return cls(strategy=strategy, trainable=trainable)
    
    @classmethod
    def from_params(cls, params: Dict[str, Any], trainable: bool = False) -> 'NormalizationLayer':
        """
        Crée une couche de normalisation à partir de paramètres.
        
        Args:
            params: Paramètres de normalisation
            trainable: Si True, les paramètres de normalisation sont apprenables
            
        Returns:
            Instance de NormalizationLayer
        """
        method = params.get('method', NormalizationMethod.MINMAX)
        
        if method == NormalizationMethod.MINMAX:
            strategy = MinMaxNormalization(
                min_val=params['min_val'],
                max_val=params['max_val'],
                target_min=params.get('target_min', 0.0),
                target_max=params.get('target_max', 1.0),
                clip_values=params.get('clip_values', True)
            )
        elif method == NormalizationMethod.ZSCORE:
            strategy = ZScoreNormalization(
                mean=params['mean'],
                std=params['std'],
                clip_sigma=params.get('clip_sigma', None)
            )
        elif method == NormalizationMethod.ROBUST:
            strategy = RobustNormalization(
                median=params['median'],
                p1=params['p1'],
                p99=params['p99'],
                target_min=params.get('target_min', -1.0),
                target_max=params.get('target_max', 1.0),
                clip_values=params.get('clip_values', True)
            )
        elif method == NormalizationMethod.BATCH:
            strategy = BatchNormStrategy(
                num_features=params['num_features'],
                eps=params.get('eps', 1e-5),
                momentum=params.get('momentum', 0.1),
                affine=params.get('affine', True),
                track_running_stats=params.get('track_running_stats', True)
            )
            
            # Si on a des statistiques enregistrées, initialise les buffers du BatchNorm
            if 'running_mean' in params and 'running_var' in params:
                strategy.batch_norm.running_mean.copy_(torch.tensor(params['running_mean']))
                strategy.batch_norm.running_var.copy_(torch.tensor(params['running_var']))
            
            # Si on a des paramètres affines, initialise les poids et biais du BatchNorm
            if 'weight' in params and 'bias' in params and strategy.batch_norm.affine:
                strategy.batch_norm.weight.copy_(torch.tensor(params['weight']))
                strategy.batch_norm.bias.copy_(torch.tensor(params['bias']))
        else:
            raise ValueError(f"Méthode de normalisation '{method}' non supportée")
        
        return cls(strategy=strategy, trainable=trainable)
    
    @classmethod
    def load(cls, file_path: str, trainable: bool = False) -> 'NormalizationLayer':
        """
        Charge une couche de normalisation à partir d'un fichier JSON.
        
        Args:
            file_path: Chemin du fichier JSON contenant les paramètres
            trainable: Si True, les paramètres de normalisation sont apprenables
            
        Returns:
            Instance de NormalizationLayer
        """
        try:
            with open(file_path, 'r') as f:
                params = json.load(f)
            
            logger.info(f"Paramètres de normalisation chargés depuis {file_path}")
            
            return cls.from_params(params, trainable=trainable)
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des paramètres de normalisation: {str(e)}")
            raise


class InputNormalization(nn.Module):
    """
    Module de normalisation des entrées pour un modèle PyTorch.
    
    Ce module permet d'intégrer la normalisation des entrées directement
    dans un modèle pour qu'elle soit appliquée automatiquement.
    """
    
    def __init__(
        self,
        norm_layer: NormalizationLayer,
        channels_last: bool = False,
        apply_to_mask: bool = False
    ):
        """
        Initialise le module de normalisation des entrées.
        
        Args:
            norm_layer: Couche de normalisation à utiliser
            channels_last: Si True, les canaux sont en dernière dimension (B, H, W, C)
            apply_to_mask: Si True, applique aussi la normalisation aux masques
        """
        super().__init__()
        self.norm_layer = norm_layer
        self.channels_last = channels_last
        self.apply_to_mask = apply_to_mask
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Normalise les entrées du modèle.
        
        Args:
            x: Tensor d'entrée à normaliser
            mask: Tensor de masque (optionnel)
            
        Returns:
            Tensor normalisé ou tuple (tensor normalisé, masque)
        """
        # Si les canaux sont en dernière dimension, réorganise les dimensions
        if self.channels_last and x.dim() > 2:
            # Déplace les canaux de la dernière à la deuxième dimension
            perm = list(range(x.dim()))
            perm.pop()  # Retire la dernière dimension
            perm.insert(1, x.dim() - 1)  # L'insère à la position 1
            x_norm = self.norm_layer(x.permute(*perm))
            
            # Restaure l'ordre des dimensions
            perm = list(range(x_norm.dim()))
            perm.pop(1)  # Retire la dimension des canaux
            perm.append(1)  # L'ajoute à la fin
            x_norm = x_norm.permute(*perm)
        else:
            x_norm = self.norm_layer(x)
        
        # Si un masque est fourni, le normalise si demandé
        if mask is not None:
            if self.apply_to_mask:
                # Si les canaux sont en dernière dimension, réorganise les dimensions
                if self.channels_last and mask.dim() > 2:
                    perm = list(range(mask.dim()))
                    perm.pop()
                    perm.insert(1, mask.dim() - 1)
                    mask_norm = self.norm_layer(mask.permute(*perm))
                    
                    perm = list(range(mask_norm.dim()))
                    perm.pop(1)
                    perm.append(1)
                    mask_norm = mask_norm.permute(*perm)
                else:
                    mask_norm = self.norm_layer(mask)
                    
                return x_norm, mask_norm
            else:
                return x_norm, mask
        
        return x_norm
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dénormalise les sorties du modèle.
        
        Args:
            x: Tensor à dénormaliser
            
        Returns:
            Tensor dénormalisé
        """
        # Si les canaux sont en dernière dimension, réorganise les dimensions
        if self.channels_last and x.dim() > 2:
            perm = list(range(x.dim()))
            perm.pop()
            perm.insert(1, x.dim() - 1)
            x_denorm = self.norm_layer.denormalize(x.permute(*perm))
            
            perm = list(range(x_denorm.dim()))
            perm.pop(1)
            perm.append(1)
            return x_denorm.permute(*perm)
        else:
            return self.norm_layer.denormalize(x)


def create_normalization_layer(
    method: Union[str, NormalizationMethod] = 'adaptive',
    stats: Optional[NormalizationStatistics] = None,
    trainable: bool = False,
    **kwargs
) -> NormalizationLayer:
    """
    Crée une couche de normalisation selon la méthode spécifiée.
    
    Args:
        method: Méthode de normalisation
        stats: Statistiques de normalisation (obligatoire pour certaines méthodes)
        trainable: Si True, les paramètres de normalisation sont apprenables
        **kwargs: Arguments supplémentaires pour la stratégie
        
    Returns:
        Couche de normalisation
    """
    if stats is not None:
        return NormalizationLayer.from_stats(stats, method=method, trainable=trainable, **kwargs)
    elif 'params' in kwargs:
        return NormalizationLayer.from_params(kwargs['params'], trainable=trainable)
    elif 'strategy' in kwargs:
        return NormalizationLayer.from_strategy(kwargs['strategy'], trainable=trainable)
    elif 'file_path' in kwargs:
        return NormalizationLayer.load(kwargs['file_path'], trainable=trainable)
    else:
        # Crée la stratégie avec les paramètres fournis
        strategy = create_normalization_strategy(method=method, **kwargs)
        return NormalizationLayer(strategy=strategy, trainable=trainable)


def normalize_batch(
    batch: torch.Tensor,
    method: Union[str, NormalizationMethod] = 'minmax',
    **kwargs
) -> torch.Tensor:
    """
    Normalise un batch de données selon la méthode spécifiée.
    
    Args:
        batch: Batch de données à normaliser [B, C, H, W] ou [B, H, W]
        method: Méthode de normalisation
        **kwargs: Arguments spécifiques à la méthode
        
    Returns:
        Batch normalisé
    """
    if method == 'minmax' or method == NormalizationMethod.MINMAX:
        # Normalisation min-max par batch
        if 'min_val' in kwargs and 'max_val' in kwargs:
            min_val = kwargs['min_val']
            max_val = kwargs['max_val']
        else:
            min_val = batch.min()
            max_val = batch.max()
        
        target_min = kwargs.get('target_min', 0.0)
        target_max = kwargs.get('target_max', 1.0)
        
        scale = (target_max - target_min) / (max_val - min_val + 1e-8)
        shift = target_min - min_val * scale
        
        return batch * scale + shift
        
    elif method == 'zscore' or method == NormalizationMethod.ZSCORE:
        # Normalisation z-score par batch
        if 'mean' in kwargs and 'std' in kwargs:
            mean = kwargs['mean']
            std = kwargs['std']
        else:
            mean = batch.mean()
            std = batch.std() + 1e-8
        
        return (batch - mean) / std
        
    elif method == 'robust' or method == NormalizationMethod.ROBUST:
        # Normalisation robuste par batch
        if all(k in kwargs for k in ['median', 'p1', 'p99']):
            median = kwargs['median']
            p1 = kwargs['p1']
            p99 = kwargs['p99']
        else:
            # Calcule approximativement les percentiles
            flat_batch = batch.flatten()
            sorted_batch, _ = torch.sort(flat_batch)
            n = len(sorted_batch)
            
            p1_idx = max(0, min(n - 1, int(0.01 * n)))
            median_idx = max(0, min(n - 1, int(0.5 * n)))
            p99_idx = max(0, min(n - 1, int(0.99 * n)))
            
            p1 = sorted_batch[p1_idx].item()
            median = sorted_batch[median_idx].item()
            p99 = sorted_batch[p99_idx].item()
        
        target_min = kwargs.get('target_min', -1.0)
        target_max = kwargs.get('target_max', 1.0)
        center = (target_max + target_min) / 2
        
        scale_low = (target_min - center) / (p1 - median + 1e-8) if p1 < median else 1.0
        scale_high = (target_max - center) / (p99 - median + 1e-8) if p99 > median else 1.0
        
        # Applique la normalisation par parties
        mask_low = batch < median
        mask_high = ~mask_low
        
        result = torch.zeros_like(batch)
        result[mask_low] = (batch[mask_low] - median) * scale_low + center
        result[mask_high] = (batch[mask_high] - median) * scale_high + center
        
        return result
        
    elif method == 'batch' or method == NormalizationMethod.BATCH:
        # Utilise la normalisation par batch de PyTorch
        batch_shape = batch.shape
        
        # Assure que le batch a 4 dimensions [B, C, H, W]
        if batch.dim() == 3:  # [B, H, W]
            batch = batch.unsqueeze(1)  # [B, 1, H, W]
            
        # Normalise par batch
        batch_mean = batch.mean(dim=(0, 2, 3), keepdim=True)
        batch_var = batch.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        batch_norm = (batch - batch_mean) / torch.sqrt(batch_var + 1e-5)
        
        # Restaure la forme d'origine
        if batch_shape != batch_norm.shape:
            batch_norm = batch_norm.reshape(batch_shape)
            
        return batch_norm
        
    else:
        raise ValueError(f"Méthode de normalisation '{method}' non supportée")


def denormalize_batch(
    batch: torch.Tensor,
    method: Union[str, NormalizationMethod] = 'minmax',
    **kwargs
) -> torch.Tensor:
    """
    Dénormalise un batch de données selon la méthode spécifiée.
    
    Args:
        batch: Batch de données à dénormaliser
        method: Méthode de normalisation utilisée
        **kwargs: Arguments spécifiques à la méthode
        
    Returns:
        Batch dénormalisé
    """
    if method == 'minmax' or method == NormalizationMethod.MINMAX:
        # Dénormalisation min-max
        if all(k in kwargs for k in ['scale', 'shift']):
            scale = kwargs['scale']
            shift = kwargs['shift']
        elif all(k in kwargs for k in ['min_val', 'max_val', 'target_min', 'target_max']):
            min_val = kwargs['min_val']
            max_val = kwargs['max_val']
            target_min = kwargs['target_min']
            target_max = kwargs['target_max']
            
            scale = (target_max - target_min) / (max_val - min_val + 1e-8)
            shift = target_min - min_val * scale
        else:
            raise ValueError("Paramètres insuffisants pour la dénormalisation min-max")
        
        return (batch - shift) / scale
        
    elif method == 'zscore' or method == NormalizationMethod.ZSCORE:
        # Dénormalisation z-score
        if 'mean' in kwargs and 'std' in kwargs:
            mean = kwargs['mean']
            std = kwargs['std']
        else:
            raise ValueError("Paramètres 'mean' et 'std' requis pour la dénormalisation z-score")
        
        return batch * std + mean
        
    elif method == 'robust' or method == NormalizationMethod.ROBUST:
        # Dénormalisation robuste
        if all(k in kwargs for k in ['median', 'p1', 'p99', 'target_min', 'target_max']):
            median = kwargs['median']
            p1 = kwargs['p1']
            p99 = kwargs['p99']
            target_min = kwargs['target_min']
            target_max = kwargs['target_max']
        else:
            raise ValueError("Paramètres insuffisants pour la dénormalisation robuste")
        
        center = (target_max + target_min) / 2
        scale_low = (target_min - center) / (p1 - median + 1e-8) if p1 < median else 1.0
        scale_high = (target_max - center) / (p99 - median + 1e-8) if p99 > median else 1.0
        
        # Applique la dénormalisation par parties
        mask_low = batch < center
        mask_high = ~mask_low
        
        result = torch.zeros_like(batch)
        result[mask_low] = (batch[mask_low] - center) / scale_low + median
        result[mask_high] = (batch[mask_high] - center) / scale_high + median
        
        return result
        
    elif method == 'batch' or method == NormalizationMethod.BATCH:
        # La dénormalisation de BatchNorm nécessite des paramètres spécifiques
        raise NotImplementedError("La dénormalisation par batch nécessite les statistiques enregistrées")
        
    else:
        raise ValueError(f"Méthode de dénormalisation '{method}' non supportée") 