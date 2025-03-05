"""
Mécanismes d'attention pour améliorer les performances des modèles de détection de trouées forestières.

Ce module contient différentes implémentations de mécanismes d'attention
qui peuvent être intégrés dans les architectures de modèles pour améliorer
leur capacité à se concentrer sur les caractéristiques importantes.
"""

from models.attention.cbam import CBAM, ChannelAttention, SpatialAttention
from models.attention.self_attention import SelfAttention

__all__ = [
    'CBAM',
    'ChannelAttention',
    'SpatialAttention',
    'SelfAttention'
] 