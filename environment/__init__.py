"""
Module de gestion d'environnement pour le projet forestgaps.

Ce module fournit des classes pour détecter et configurer automatiquement
l'environnement d'exécution (Colab ou local).
"""

import sys
from .base import Environment
from .local import LocalEnvironment
from .colab import ColabEnvironment


def detect_environment():
    """
    Détecte automatiquement l'environnement d'exécution.
    
    Returns:
        Une instance de l'environnement détecté (ColabEnvironment ou LocalEnvironment).
    """
    return Environment.detect()


def setup_environment():
    """
    Détecte et configure automatiquement l'environnement d'exécution.
    
    Returns:
        Une instance de l'environnement configuré.
    """
    env = detect_environment()
    env.setup()
    return env


def get_device():
    """
    Détecte et renvoie le dispositif à utiliser pour les calculs (CPU ou GPU).
    
    Returns:
        Le dispositif à utiliser ('cuda' ou 'cpu').
    """
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


# Exporter les classes et fonctions principales
__all__ = [
    'Environment',
    'LocalEnvironment',
    'ColabEnvironment',
    'detect_environment',
    'setup_environment',
    'get_device'
]
