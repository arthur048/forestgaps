"""
Module de configuration pour le projet forestgaps.

Ce module fournit des classes et des fonctions pour gérer les configurations
du projet, notamment pour le traitement des données, les modèles et l'entraînement.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from .base import config
from .schema import validate_config, DataSchema, ModelSchema, TrainingSchema


def load_default_config() -> Config:
    """
    Charge la configuration par défaut à partir des fichiers YAML.
    
    Returns:
        Une instance de Config avec les paramètres par défaut.
    """
    config = Config()
    
    # Chemins des fichiers de configuration par défaut
    config_dir = os.path.dirname(os.path.abspath(__file__))
    data_config_path = os.path.join(config_dir, 'defaults', 'data.yaml')
    models_config_path = os.path.join(config_dir, 'defaults', 'models.yaml')
    training_config_path = os.path.join(config_dir, 'defaults', 'training.yaml')
    
    # Charger les configurations par défaut
    config.merge_configs(data_config_path, models_config_path, training_config_path)
    
    # Créer les répertoires spécifiques
    config.TILES_DIR = os.path.join(config.PROCESSED_DIR, 'tiles')
    config.TRAIN_TILES_DIR = os.path.join(config.TILES_DIR, 'train')
    config.VAL_TILES_DIR = os.path.join(config.TILES_DIR, 'val')
    config.TEST_TILES_DIR = os.path.join(config.TILES_DIR, 'test')
    config.DATA_EXTERNAL_TEST_DIR = os.path.join(config.DATA_DIR, 'external_test')
    
    # Créer les répertoires pour les modèles
    config.UNET_DIR = os.path.join(config.MODELS_DIR, 'unet')
    config.CHECKPOINTS_DIR = os.path.join(config.UNET_DIR, 'checkpoints')
    config.LOGS_DIR = os.path.join(config.UNET_DIR, 'logs')
    config.RESULTS_DIR = os.path.join(config.UNET_DIR, 'results')
    config.VISUALIZATIONS_DIR = os.path.join(config.UNET_DIR, 'visualizations')
    
    # Créer les répertoires nécessaires
    directories = [
        config.DATA_DIR, config.PROCESSED_DIR, config.MODELS_DIR,
        config.TILES_DIR, config.TRAIN_TILES_DIR, config.VAL_TILES_DIR, config.TEST_TILES_DIR,
        config.DATA_EXTERNAL_TEST_DIR, config.UNET_DIR, config.CHECKPOINTS_DIR,
        config.LOGS_DIR, config.RESULTS_DIR, config.VISUALIZATIONS_DIR
    ]
    config.create_directories(directories)
    
    return config


def load_config_from_file(config_path: str) -> Config:
    """
    Charge une configuration à partir d'un fichier.
    
    Args:
        config_path: Chemin vers le fichier de configuration.
        
    Returns:
        Une instance de Config avec les paramètres chargés.
    """
    config = Config(config_path)
    return config


def create_config_from_dict(config_dict: Dict[str, Any]) -> Config:
    """
    Crée une configuration à partir d'un dictionnaire.
    
    Args:
        config_dict: Dictionnaire contenant les paramètres de configuration.
        
    Returns:
        Une instance de Config avec les paramètres spécifiés.
    """
    # Valider la configuration
    validated_config = validate_config(config_dict)
    
    # Créer une nouvelle configuration
    config = Config()
    config.update_from_dict(validated_config)
    
    return config


# Exporter les classes et fonctions principales
__all__ = [
    'Config',
    'validate_config',
    'DataSchema',
    'ModelSchema',
    'TrainingSchema',
    'load_default_config',
    'load_config_from_file',
    'create_config_from_dict'
]
