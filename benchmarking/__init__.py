"""
Module de benchmarking pour la comparaison des modèles ForestGaps-DL.

Ce module fournit des outils pour comparer systématiquement différentes 
architectures de modèles et configurations d'entraînement pour la 
détection des trouées forestières.
"""

from benchmarking.comparison import ModelComparison
from benchmarking.metrics import AggregatedMetrics, MetricsTracker
from benchmarking.visualization import BenchmarkVisualizer
from benchmarking.reporting import BenchmarkReporter

__all__ = [
    'ModelComparison',
    'AggregatedMetrics',
    'MetricsTracker',
    'BenchmarkVisualizer',
    'BenchmarkReporter'
] 