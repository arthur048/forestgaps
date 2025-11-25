"""
Module de stratégies de normalisation adaptatives.

Ce module fournit différentes stratégies de normalisation pour les données
de détection de trouées forestières, avec des approches adaptatives selon
les caractéristiques des données.
"""

import logging
import abc
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

from forestgaps.data.normalization.statistics import NormalizationStatistics

# Configuration du logger
logger = logging.getLogger(__name__)


class NormalizationMethod(str, Enum):
    """Énumération des méthodes de normalisation supportées."""
    MINMAX = "minmax"
    ZSCORE = "zscore"
    ROBUST = "robust" 
    PERCENTILE = "percentile"
    BATCH = "batch"
    INSTANCE = "instance"
    GROUP = "group"
    LAYER = "layer"


class NormalizationStrategy(abc.ABC):
    """
    Classe abstraite pour les stratégies de normalisation.
    
    Toutes les stratégies de normalisation doivent implémenter cette interface
    pour assurer une utilisation cohérente dans le pipeline.
    """
    
    @abc.abstractmethod
    def normalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalise les données selon la stratégie.
        
        Args:
            data: Données à normaliser (NumPy array ou Tensor PyTorch)
            
        Returns:
            Données normalisées (même type que l'entrée)
        """
        pass
    
    @abc.abstractmethod
    def denormalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Dénormalise les données selon la stratégie inverse.
        
        Args:
            data: Données à dénormaliser (NumPy array ou Tensor PyTorch)
            
        Returns:
            Données dénormalisées (même type que l'entrée)
        """
        pass
    
    @abc.abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Retourne les paramètres de la stratégie de normalisation.
        
        Returns:
            Dictionnaire contenant les paramètres de la stratégie
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def from_stats(cls, stats: NormalizationStatistics, **kwargs) -> 'NormalizationStrategy':
        """
        Crée une stratégie de normalisation à partir de statistiques.
        
        Args:
            stats: Statistiques de normalisation
            **kwargs: Arguments supplémentaires spécifiques à la stratégie
            
        Returns:
            Instance de la stratégie de normalisation
        """
        pass


class MinMaxNormalization(NormalizationStrategy):
    """
    Stratégie de normalisation Min-Max.
    
    Cette stratégie normalise les données dans une plage spécifiée (par défaut [0, 1])
    en utilisant les valeurs minimales et maximales.
    """
    
    def __init__(
        self,
        min_val: float,
        max_val: float,
        target_min: float = 0.0,
        target_max: float = 1.0,
        clip_values: bool = True
    ):
        """
        Initialise la stratégie de normalisation Min-Max.
        
        Args:
            min_val: Valeur minimale des données
            max_val: Valeur maximale des données
            target_min: Valeur minimale cible après normalisation
            target_max: Valeur maximale cible après normalisation
            clip_values: Écrête les valeurs en dehors de la plage [min_val, max_val]
        """
        self.min_val = min_val
        self.max_val = max_val
        self.target_min = target_min
        self.target_max = target_max
        self.clip_values = clip_values
        
        # Calcule le facteur d'échelle et le décalage
        self.scale = (target_max - target_min) / (max_val - min_val) if max_val > min_val else 1.0
        self.shift = target_min - min_val * self.scale
    
    def normalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalise les données selon la stratégie Min-Max.
        
        Args:
            data: Données à normaliser
            
        Returns:
            Données normalisées
        """
        is_torch = isinstance(data, torch.Tensor)
        
        # Convertit en NumPy si nécessaire
        if is_torch:
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data
            
        # Écrête les valeurs si demandé
        if self.clip_values:
            data_np = np.clip(data_np, self.min_val, self.max_val)
        
        # Applique la normalisation
        normalized = data_np * self.scale + self.shift
        
        # Reconvertit en Tensor si nécessaire
        if is_torch:
            if data.is_cuda:
                return torch.tensor(normalized, device=data.device, dtype=data.dtype)
            else:
                return torch.tensor(normalized, dtype=data.dtype)
        else:
            return normalized
    
    def denormalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Dénormalise les données selon la stratégie Min-Max inverse.
        
        Args:
            data: Données à dénormaliser
            
        Returns:
            Données dénormalisées
        """
        is_torch = isinstance(data, torch.Tensor)
        
        # Convertit en NumPy si nécessaire
        if is_torch:
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data
        
        # Applique la dénormalisation
        denormalized = (data_np - self.shift) / self.scale
        
        # Écrête les valeurs si demandé
        if self.clip_values:
            denormalized = np.clip(denormalized, self.min_val, self.max_val)
        
        # Reconvertit en Tensor si nécessaire
        if is_torch:
            if data.is_cuda:
                return torch.tensor(denormalized, device=data.device, dtype=data.dtype)
            else:
                return torch.tensor(denormalized, dtype=data.dtype)
        else:
            return denormalized
    
    def get_params(self) -> Dict[str, Any]:
        """
        Retourne les paramètres de la stratégie Min-Max.
        
        Returns:
            Dictionnaire contenant les paramètres
        """
        return {
            'method': NormalizationMethod.MINMAX,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'target_min': self.target_min,
            'target_max': self.target_max,
            'clip_values': self.clip_values,
            'scale': self.scale,
            'shift': self.shift
        }
    
    @classmethod
    def from_stats(cls, stats: NormalizationStatistics, **kwargs) -> 'MinMaxNormalization':
        """
        Crée une stratégie Min-Max à partir de statistiques.
        
        Args:
            stats: Statistiques de normalisation
            **kwargs: Arguments supplémentaires (target_min, target_max, clip_values)
            
        Returns:
            Instance de MinMaxNormalization
        """
        params = stats.get_normalization_params('minmax')
        
        return cls(
            min_val=params['min'],
            max_val=params['max'],
            target_min=kwargs.get('target_min', 0.0),
            target_max=kwargs.get('target_max', 1.0),
            clip_values=kwargs.get('clip_values', True)
        )


class ZScoreNormalization(NormalizationStrategy):
    """
    Stratégie de normalisation Z-Score.
    
    Cette stratégie normalise les données en soustrayant la moyenne et
    en divisant par l'écart-type pour obtenir une distribution centrée
    sur 0 avec un écart-type de 1.
    """
    
    def __init__(
        self,
        mean: float,
        std: float,
        epsilon: float = 1e-8,
        clip_sigma: Optional[float] = None
    ):
        """
        Initialise la stratégie de normalisation Z-Score.
        
        Args:
            mean: Moyenne des données
            std: Écart-type des données
            epsilon: Valeur minimale pour éviter la division par zéro
            clip_sigma: Écrête les valeurs au-delà de N sigmas (None = pas d'écrêtage)
        """
        self.mean = mean
        self.std = std if std > epsilon else epsilon
        self.epsilon = epsilon
        self.clip_sigma = clip_sigma
    
    def normalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalise les données selon la stratégie Z-Score.
        
        Args:
            data: Données à normaliser
            
        Returns:
            Données normalisées
        """
        is_torch = isinstance(data, torch.Tensor)
        
        # Normalisation
        if is_torch:
            normalized = (data - self.mean) / self.std
            
            # Écrête les valeurs si demandé
            if self.clip_sigma is not None:
                normalized = torch.clamp(normalized, -self.clip_sigma, self.clip_sigma)
        else:
            normalized = (data - self.mean) / self.std
            
            # Écrête les valeurs si demandé
            if self.clip_sigma is not None:
                normalized = np.clip(normalized, -self.clip_sigma, self.clip_sigma)
        
        return normalized
    
    def denormalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Dénormalise les données selon la stratégie Z-Score inverse.
        
        Args:
            data: Données à dénormaliser
            
        Returns:
            Données dénormalisées
        """
        is_torch = isinstance(data, torch.Tensor)
        
        # Dénormalisation
        if is_torch:
            return data * self.std + self.mean
        else:
            return data * self.std + self.mean
    
    def get_params(self) -> Dict[str, Any]:
        """
        Retourne les paramètres de la stratégie Z-Score.
        
        Returns:
            Dictionnaire contenant les paramètres
        """
        return {
            'method': NormalizationMethod.ZSCORE,
            'mean': self.mean,
            'std': self.std,
            'epsilon': self.epsilon,
            'clip_sigma': self.clip_sigma
        }
    
    @classmethod
    def from_stats(cls, stats: NormalizationStatistics, **kwargs) -> 'ZScoreNormalization':
        """
        Crée une stratégie Z-Score à partir de statistiques.
        
        Args:
            stats: Statistiques de normalisation
            **kwargs: Arguments supplémentaires (epsilon, clip_sigma)
            
        Returns:
            Instance de ZScoreNormalization
        """
        params = stats.get_normalization_params('zscore')
        
        return cls(
            mean=params['mean'],
            std=params['std'],
            epsilon=kwargs.get('epsilon', 1e-8),
            clip_sigma=kwargs.get('clip_sigma', None)
        )


class RobustNormalization(NormalizationStrategy):
    """
    Stratégie de normalisation robuste.
    
    Cette stratégie utilise la médiane et les percentiles pour normaliser
    les données, ce qui est plus robuste aux valeurs aberrantes.
    """
    
    def __init__(
        self,
        median: float,
        p1: float,
        p99: float,
        target_min: float = -1.0,
        target_max: float = 1.0,
        clip_values: bool = True
    ):
        """
        Initialise la stratégie de normalisation robuste.
        
        Args:
            median: Médiane des données
            p1: 1er percentile des données
            p99: 99ème percentile des données
            target_min: Valeur minimale cible après normalisation
            target_max: Valeur maximale cible après normalisation
            clip_values: Écrête les valeurs en dehors de la plage [p1, p99]
        """
        self.median = median
        self.p1 = p1
        self.p99 = p99
        self.target_min = target_min
        self.target_max = target_max
        self.clip_values = clip_values
        
        # Calcule les facteurs d'échelle pour les valeurs inférieures et supérieures à la médiane
        self.scale_low = (target_min - (target_max + target_min) / 2) / (p1 - median) if p1 < median else 1.0
        self.scale_high = (target_max - (target_max + target_min) / 2) / (p99 - median) if p99 > median else 1.0
        self.center = (target_max + target_min) / 2
    
    def normalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalise les données selon la stratégie robuste.
        
        Args:
            data: Données à normaliser
            
        Returns:
            Données normalisées
        """
        is_torch = isinstance(data, torch.Tensor)
        
        # Convertit en NumPy si nécessaire
        if is_torch:
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data
            
        # Écrête les valeurs si demandé
        if self.clip_values:
            data_np = np.clip(data_np, self.p1, self.p99)
        
        # Crée un masque pour les valeurs inférieures et supérieures à la médiane
        mask_low = data_np < self.median
        mask_high = ~mask_low
        
        # Initialise le tableau normalisé
        normalized = np.zeros_like(data_np)
        
        # Applique la normalisation pour les valeurs inférieures à la médiane
        normalized[mask_low] = (data_np[mask_low] - self.median) * self.scale_low + self.center
        
        # Applique la normalisation pour les valeurs supérieures à la médiane
        normalized[mask_high] = (data_np[mask_high] - self.median) * self.scale_high + self.center
        
        # Reconvertit en Tensor si nécessaire
        if is_torch:
            if data.is_cuda:
                return torch.tensor(normalized, device=data.device, dtype=data.dtype)
            else:
                return torch.tensor(normalized, dtype=data.dtype)
        else:
            return normalized
    
    def denormalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Dénormalise les données selon la stratégie robuste inverse.
        
        Args:
            data: Données à dénormaliser
            
        Returns:
            Données dénormalisées
        """
        is_torch = isinstance(data, torch.Tensor)
        
        # Convertit en NumPy si nécessaire
        if is_torch:
            data_np = data.detach().cpu().numpy()
        else:
            data_np = data
        
        # Crée un masque pour les valeurs inférieures et supérieures au centre
        mask_low = data_np < self.center
        mask_high = ~mask_low
        
        # Initialise le tableau dénormalisé
        denormalized = np.zeros_like(data_np)
        
        # Applique la dénormalisation pour les valeurs inférieures au centre
        denormalized[mask_low] = (data_np[mask_low] - self.center) / self.scale_low + self.median
        
        # Applique la dénormalisation pour les valeurs supérieures au centre
        denormalized[mask_high] = (data_np[mask_high] - self.center) / self.scale_high + self.median
        
        # Écrête les valeurs si demandé
        if self.clip_values:
            denormalized = np.clip(denormalized, self.p1, self.p99)
        
        # Reconvertit en Tensor si nécessaire
        if is_torch:
            if data.is_cuda:
                return torch.tensor(denormalized, device=data.device, dtype=data.dtype)
            else:
                return torch.tensor(denormalized, dtype=data.dtype)
        else:
            return denormalized
    
    def get_params(self) -> Dict[str, Any]:
        """
        Retourne les paramètres de la stratégie robuste.
        
        Returns:
            Dictionnaire contenant les paramètres
        """
        return {
            'method': NormalizationMethod.ROBUST,
            'median': self.median,
            'p1': self.p1,
            'p99': self.p99,
            'target_min': self.target_min,
            'target_max': self.target_max,
            'clip_values': self.clip_values,
            'scale_low': self.scale_low,
            'scale_high': self.scale_high,
            'center': self.center
        }
    
    @classmethod
    def from_stats(cls, stats: NormalizationStatistics, **kwargs) -> 'RobustNormalization':
        """
        Crée une stratégie robuste à partir de statistiques.
        
        Args:
            stats: Statistiques de normalisation
            **kwargs: Arguments supplémentaires (target_min, target_max, clip_values)
            
        Returns:
            Instance de RobustNormalization
        """
        params = stats.get_normalization_params('robust')
        
        return cls(
            median=params['median'],
            p1=params['p1'],
            p99=params['p99'],
            target_min=kwargs.get('target_min', -1.0),
            target_max=kwargs.get('target_max', 1.0),
            clip_values=kwargs.get('clip_values', True)
        )


class AdaptiveNormalization(NormalizationStrategy):
    """
    Stratégie de normalisation adaptative.
    
    Cette stratégie choisit automatiquement la méthode de normalisation
    la plus appropriée en fonction des caractéristiques des données.
    """
    
    def __init__(
        self,
        stats: NormalizationStatistics,
        skew_threshold: float = 1.0,
        kurtosis_threshold: float = 3.0,
        outlier_percentage_threshold: float = 0.05
    ):
        """
        Initialise la stratégie de normalisation adaptative.
        
        Args:
            stats: Statistiques de normalisation
            skew_threshold: Seuil d'asymétrie au-delà duquel utiliser une normalisation robuste
            kurtosis_threshold: Seuil de kurtosis au-delà duquel utiliser une normalisation robuste
            outlier_percentage_threshold: Seuil de pourcentage de valeurs aberrantes
        """
        self.stats = stats
        self.skew_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.outlier_percentage_threshold = outlier_percentage_threshold
        
        # Calcule les caractéristiques des données
        global_stats = stats.stats['global']
        self.mean = global_stats['mean']
        self.std = global_stats['std']
        self.min_val = global_stats['min']
        self.max_val = global_stats['max']
        self.median = global_stats['median']
        self.p1 = global_stats['p1']
        self.p99 = global_stats['p99']
        
        # Calcule les indicateurs pour choisir la méthode
        self.range = self.max_val - self.min_val
        self.lower_range = self.median - self.min_val
        self.upper_range = self.max_val - self.median
        self.range_ratio = max(self.lower_range, self.upper_range) / (min(self.lower_range, self.upper_range) + 1e-8)
        
        # Indicateur d'asymétrie (approximatif)
        self.skewness = abs((self.mean - self.median) / (self.std + 1e-8))
        
        # Pourcentage approximatif de valeurs aberrantes
        iqr = self.p99 - self.p1
        lower_bound = self.p1 - 1.5 * iqr
        upper_bound = self.p99 + 1.5 * iqr
        self.outlier_percentage = max(
            (self.min_val < lower_bound) * (lower_bound - self.min_val) / (self.range + 1e-8),
            (self.max_val > upper_bound) * (self.max_val - upper_bound) / (self.range + 1e-8)
        )
        
        # Choisit la méthode de normalisation
        if (self.skewness > skew_threshold or 
            self.range_ratio > 5.0 or 
            self.outlier_percentage > outlier_percentage_threshold):
            self.method = NormalizationMethod.ROBUST
            self.strategy = RobustNormalization(
                median=self.median,
                p1=self.p1,
                p99=self.p99
            )
        else:
            self.method = NormalizationMethod.ZSCORE
            self.strategy = ZScoreNormalization(
                mean=self.mean,
                std=self.std
            )
        
        logger.info(f"Stratégie de normalisation adaptative choisie: {self.method}")
        logger.info(f"Indicateurs: skewness={self.skewness:.2f}, range_ratio={self.range_ratio:.2f}, "
                    f"outlier_percentage={self.outlier_percentage:.2f}")
    
    def normalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalise les données selon la stratégie adaptative choisie.
        
        Args:
            data: Données à normaliser
            
        Returns:
            Données normalisées
        """
        return self.strategy.normalize(data)
    
    def denormalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Dénormalise les données selon la stratégie adaptative inverse.
        
        Args:
            data: Données à dénormaliser
            
        Returns:
            Données dénormalisées
        """
        return self.strategy.denormalize(data)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Retourne les paramètres de la stratégie adaptative.
        
        Returns:
            Dictionnaire contenant les paramètres
        """
        return {
            'method': 'adaptive',
            'selected_method': self.method,
            'strategy_params': self.strategy.get_params(),
            'indicators': {
                'skewness': self.skewness,
                'range_ratio': self.range_ratio,
                'outlier_percentage': self.outlier_percentage
            }
        }
    
    @classmethod
    def from_stats(cls, stats: NormalizationStatistics, **kwargs) -> 'AdaptiveNormalization':
        """
        Crée une stratégie adaptative à partir de statistiques.
        
        Args:
            stats: Statistiques de normalisation
            **kwargs: Arguments supplémentaires (seuils)
            
        Returns:
            Instance de AdaptiveNormalization
        """
        return cls(
            stats=stats,
            skew_threshold=kwargs.get('skew_threshold', 1.0),
            kurtosis_threshold=kwargs.get('kurtosis_threshold', 3.0),
            outlier_percentage_threshold=kwargs.get('outlier_percentage_threshold', 0.05)
        )


class BatchNormStrategy(NormalizationStrategy):
    """
    Stratégie de normalisation utilisant la normalisation par batch (BatchNorm).
    
    Cette stratégie est principalement utilisée pendant l'entraînement et
    s'intègre avec les modules PyTorch.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialise la stratégie de normalisation par batch.
        
        Args:
            num_features: Nombre de canaux des données
            eps: Valeur epsilon pour la stabilité numérique
            momentum: Momentum pour les statistiques de moyenne/variance mobiles
            affine: Si True, utilise des paramètres affines apprenables
            track_running_stats: Si True, suit les statistiques globales
            device: Périphérique sur lequel placer les tenseurs
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        
        # Crée le module BatchNorm
        self.batch_norm = nn.BatchNorm2d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )
        
        if device is not None:
            self.batch_norm.to(device)
    
    def normalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalise les données avec BatchNorm.
        
        Args:
            data: Données à normaliser (doit être un tensor 4D [B, C, H, W])
            
        Returns:
            Données normalisées
        """
        is_numpy = isinstance(data, np.ndarray)
        is_training = self.batch_norm.training
        
        # Convertit en Tensor si nécessaire
        if is_numpy:
            data_tensor = torch.tensor(data, dtype=torch.float32)
            if self.device is not None:
                data_tensor = data_tensor.to(self.device)
        else:
            data_tensor = data
        
        # S'assure que le tensor a 4 dimensions [B, C, H, W]
        if data_tensor.dim() == 2:  # [H, W]
            data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif data_tensor.dim() == 3:  # [C, H, W] ou [B, H, W]
            if data_tensor.shape[0] == self.num_features:  # [C, H, W]
                data_tensor = data_tensor.unsqueeze(0)  # [1, C, H, W]
            else:  # [B, H, W]
                data_tensor = data_tensor.unsqueeze(1)  # [B, 1, H, W]
        
        # Applique la normalisation par batch
        normalized = self.batch_norm(data_tensor)
        
        # Restaure les dimensions d'origine
        if is_numpy:
            return normalized.cpu().numpy()
        else:
            return normalized
    
    def denormalize(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Dénormalise les données (approximation, car BatchNorm n'est pas facilement inversible).
        
        Args:
            data: Données à dénormaliser
            
        Returns:
            Données dénormalisées (approximation)
        """
        is_numpy = isinstance(data, np.ndarray)
        
        # Convertit en Tensor si nécessaire
        if is_numpy:
            data_tensor = torch.tensor(data, dtype=torch.float32)
            if self.device is not None:
                data_tensor = data_tensor.to(self.device)
        else:
            data_tensor = data
        
        # S'assure que le tensor a 4 dimensions [B, C, H, W]
        original_shape = data_tensor.shape
        if data_tensor.dim() == 2:  # [H, W]
            data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif data_tensor.dim() == 3:  # [C, H, W] ou [B, H, W]
            if data_tensor.shape[0] == self.num_features:  # [C, H, W]
                data_tensor = data_tensor.unsqueeze(0)  # [1, C, H, W]
            else:  # [B, H, W]
                data_tensor = data_tensor.unsqueeze(1)  # [B, 1, H, W]
        
        # Approximation de la dénormalisation (inverse approximatif de BatchNorm)
        if self.affine:
            # Si affine=True, on peut utiliser les paramètres gamma et beta
            gamma = self.batch_norm.weight.view(1, -1, 1, 1)
            beta = self.batch_norm.bias.view(1, -1, 1, 1)
            
            if self.track_running_stats:
                # Utilise les statistiques enregistrées
                mean = self.batch_norm.running_mean.view(1, -1, 1, 1)
                var = self.batch_norm.running_var.view(1, -1, 1, 1)
                
                # On inverse la normalisation: x = (x_norm * gamma + beta) * sqrt(var + eps) + mean
                denormalized = (data_tensor - beta) / gamma * torch.sqrt(var + self.eps) + mean
            else:
                # Sans statistiques, on ne peut qu'appliquer l'inverse de la transformation affine
                denormalized = (data_tensor - beta) / gamma
        else:
            # Sans transformation affine, on ne peut pas vraiment dénormaliser
            logger.warning("Dénormalisation de BatchNorm non affine n'est qu'une approximation")
            denormalized = data_tensor
        
        # Restaure les dimensions d'origine
        if len(original_shape) < 4:
            denormalized = denormalized.reshape(original_shape)
        
        if is_numpy:
            return denormalized.cpu().numpy()
        else:
            return denormalized
    
    def get_params(self) -> Dict[str, Any]:
        """
        Retourne les paramètres de la stratégie BatchNorm.
        
        Returns:
            Dictionnaire contenant les paramètres
        """
        params = {
            'method': NormalizationMethod.BATCH,
            'num_features': self.num_features,
            'eps': self.eps,
            'momentum': self.momentum,
            'affine': self.affine,
            'track_running_stats': self.track_running_stats
        }
        
        # Ajoute les statistiques courantes si disponibles
        if self.track_running_stats:
            params['running_mean'] = self.batch_norm.running_mean.cpu().numpy().tolist()
            params['running_var'] = self.batch_norm.running_var.cpu().numpy().tolist()
        
        # Ajoute les paramètres affines si disponibles
        if self.affine:
            params['weight'] = self.batch_norm.weight.cpu().numpy().tolist()
            params['bias'] = self.batch_norm.bias.cpu().numpy().tolist()
        
        return params
    
    @classmethod
    def from_stats(cls, stats: NormalizationStatistics, **kwargs) -> 'BatchNormStrategy':
        """
        Crée une stratégie BatchNorm à partir de statistiques.
        
        Args:
            stats: Statistiques de normalisation (non utilisées directement)
            **kwargs: Arguments obligatoires (num_features) et optionnels
            
        Returns:
            Instance de BatchNormStrategy
        """
        if 'num_features' not in kwargs:
            raise ValueError("Le paramètre 'num_features' est obligatoire pour BatchNormStrategy")
        
        return cls(
            num_features=kwargs['num_features'],
            eps=kwargs.get('eps', 1e-5),
            momentum=kwargs.get('momentum', 0.1),
            affine=kwargs.get('affine', True),
            track_running_stats=kwargs.get('track_running_stats', True),
            device=kwargs.get('device', None)
        )


def create_normalization_strategy(
    method: Union[str, NormalizationMethod],
    stats: Optional[NormalizationStatistics] = None,
    **kwargs
) -> NormalizationStrategy:
    """
    Crée une stratégie de normalisation selon la méthode spécifiée.
    
    Args:
        method: Méthode de normalisation ('minmax', 'zscore', 'robust', 'adaptive', etc.)
        stats: Statistiques de normalisation (obligatoire pour certaines méthodes)
        **kwargs: Arguments supplémentaires spécifiques à la méthode
        
    Returns:
        Instance de la stratégie de normalisation
    """
    if isinstance(method, str):
        try:
            method = NormalizationMethod(method.lower())
        except ValueError:
            raise ValueError(f"Méthode de normalisation '{method}' non supportée")
    
    if method == NormalizationMethod.MINMAX:
        if stats is not None:
            return MinMaxNormalization.from_stats(stats, **kwargs)
        elif all(k in kwargs for k in ['min_val', 'max_val']):
            return MinMaxNormalization(**kwargs)
        else:
            raise ValueError("Statistiques ou paramètres min_val/max_val requis pour la normalisation MinMax")
    
    elif method == NormalizationMethod.ZSCORE:
        if stats is not None:
            return ZScoreNormalization.from_stats(stats, **kwargs)
        elif all(k in kwargs for k in ['mean', 'std']):
            return ZScoreNormalization(**kwargs)
        else:
            raise ValueError("Statistiques ou paramètres mean/std requis pour la normalisation ZScore")
    
    elif method == NormalizationMethod.ROBUST:
        if stats is not None:
            return RobustNormalization.from_stats(stats, **kwargs)
        elif all(k in kwargs for k in ['median', 'p1', 'p99']):
            return RobustNormalization(**kwargs)
        else:
            raise ValueError("Statistiques ou paramètres median/p1/p99 requis pour la normalisation Robust")
    
    elif method == NormalizationMethod.PERCENTILE:
        # Implémentation simplifiée, utilise RobustNormalization en interne
        if stats is not None:
            return RobustNormalization.from_stats(stats, **kwargs)
        else:
            raise ValueError("Statistiques requises pour la normalisation Percentile")
    
    elif method == NormalizationMethod.BATCH:
        return BatchNormStrategy.from_stats(stats, **kwargs)
    
    elif method == NormalizationMethod.INSTANCE:
        # Non implémenté directement, renvoie BatchNormStrategy avec num_features=1
        kwargs['num_features'] = kwargs.get('num_features', 1)
        return BatchNormStrategy.from_stats(stats, **kwargs)
    
    elif method == NormalizationMethod.GROUP:
        # Non implémenté directement
        raise NotImplementedError("La normalisation de groupe n'est pas encore implémentée")
    
    elif method == NormalizationMethod.LAYER:
        # Non implémenté directement
        raise NotImplementedError("La normalisation de couche n'est pas encore implémentée")
    
    elif method == 'adaptive':
        if stats is not None:
            return AdaptiveNormalization.from_stats(stats, **kwargs)
        else:
            raise ValueError("Statistiques requises pour la normalisation Adaptive")
    
    else:
        raise ValueError(f"Méthode de normalisation '{method}' non supportée") 