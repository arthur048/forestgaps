"""
Feature-wise Linear Modulation (FiLM) pour les modèles de détection de trouées forestières.

Ce module implémente les couches FiLM qui permettent de conditionner les caractéristiques
en fonction de paramètres externes, ce qui peut améliorer la flexibilité et
la performance des modèles pour différentes conditions géographiques ou saisonnières.
"""

from models.film.layers import FiLMLayer, FiLMGenerator
from models.film.blocks import FiLMBlock, FiLMResidualBlock

__all__ = [
    'FiLMLayer',
    'FiLMGenerator',
    'FiLMBlock',
    'FiLMResidualBlock'
] 