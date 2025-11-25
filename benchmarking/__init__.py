"""
Module de benchmarking pour la comparaison des modèles ForestGaps.

Ce module fournit des outils pour comparer systématiquement différentes 
architectures de modèles et configurations d'entraînement pour la 
détection des trouées forestières.
"""

from .comparison import ModelComparison
from .metrics import AggregatedMetrics, MetricsTracker
from .visualization import BenchmarkVisualizer
from .reporting import BenchmarkReporter

__all__ = [
    'ModelComparison',
    'AggregatedMetrics',
    'MetricsTracker',
    'BenchmarkVisualizer',
    'BenchmarkReporter'
] 