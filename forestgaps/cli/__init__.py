"""
Module d'interface en ligne de commande pour ForestGaps.

Ce module fournit des interfaces en ligne de commande pour les différentes
fonctionnalités du workflow ForestGaps.
"""

from . import preprocessing_cli
from . import training_cli
from . import data
from . import train
from . import evaluate

__all__ = ['preprocessing_cli', 'training_cli', 'data', 'train', 'evaluate']
