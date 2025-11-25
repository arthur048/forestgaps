"""
Module de normalisation des données pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour normaliser les données raster (DSM/CHM)
avant l'entraînement des modèles de détection des trouées forestières.
"""

from .statistics import (
    NormalizationStatistics,
    compute_normalization_statistics,
    batch_compute_statistics
)

from .strategies import (
    NormalizationMethod,
    NormalizationStrategy,
    MinMaxNormalization,
    ZScoreNormalization,
    RobustNormalization,
    AdaptiveNormalization,
    BatchNormStrategy,
    create_normalization_strategy
)

from .normalization import (
    NormalizationLayer,
    InputNormalization,
    create_normalization_layer,
    normalize_batch,
    denormalize_batch
)

from .io import (
    save_stats_json,
    load_stats_json,
    save_stats_pickle,
    load_stats_pickle,
    stats_to_dataframe,
    stats_to_csv,
    plot_stats_histogram,
    generate_stats_report,
    compare_stats,
    merge_stats,
    export_stats_to_onnx
)

__all__ = [
    # Statistiques
    'NormalizationStatistics',
    'compute_normalization_statistics',
    'batch_compute_statistics',
    
    # Stratégies
    'NormalizationMethod',
    'NormalizationStrategy',
    'MinMaxNormalization',
    'ZScoreNormalization',
    'RobustNormalization',
    'AdaptiveNormalization',
    'BatchNormStrategy',
    'create_normalization_strategy',
    
    # Couches de normalisation
    'NormalizationLayer',
    'InputNormalization',
    'create_normalization_layer',
    'normalize_batch',
    'denormalize_batch',
    
    # I/O
    'save_stats_json',
    'load_stats_json',
    'save_stats_pickle',
    'load_stats_pickle',
    'stats_to_dataframe',
    'stats_to_csv',
    'plot_stats_histogram',
    'generate_stats_report',
    'compare_stats',
    'merge_stats',
    'export_stats_to_onnx'
] 