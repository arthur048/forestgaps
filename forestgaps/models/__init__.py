"""
Module pour les modèles de détection des trouées forestières.

Ce module fournit des implémentations de divers modèles de segmentation
pour la détection des trouées forestières à partir d'images de hauteur
de canopée ou de modèles numériques de surface.
"""

import logging
from typing import Dict, Any, Type, Optional

from .registry import ModelRegistry, get_model_from_config, model_registry
from .base import (
    ForestGapModel,
    ThresholdConditionedModel,
    UNetBaseModel,
    ThresholdConditionedUNet
)

# Configuration du logging
logger = logging.getLogger(__name__)

# Exposer les fonctions principales
__all__ = [
    # Classes de base
    "ForestGapModel",
    "ThresholdConditionedModel",
    "UNetBaseModel",
    "ThresholdConditionedUNet",
    
    # Registre de modèles
    "ModelRegistry",
    "model_registry",
    "get_model_from_config",
    
    # Fonctions utilitaires
    "create_model",
    "list_available_models",
]


def create_model(model_name: str, **kwargs) -> ForestGapModel:
    """
    Crée une instance d'un modèle à partir de son nom.
    
    Args:
        model_name: Nom du modèle à créer
        **kwargs: Arguments spécifiques au modèle
        
    Returns:
        Instance du modèle
        
    Raises:
        ValueError: Si le modèle n'est pas trouvé dans le registre
    """
    logger.info(f"Création d'un modèle de type '{model_name}'")
    return model_registry.create(model_name, **kwargs)


def list_available_models() -> Dict[str, Type[ForestGapModel]]:
    """
    Liste tous les modèles disponibles dans le registre.
    
    Returns:
        Dictionnaire des noms de modèles et leurs classes
    """
    return model_registry.list_models()


# Import des implémentations spécifiques
# Ces imports doivent être à la fin pour éviter les imports circulaires
# et permettre l'enregistrement des modèles dans le registre

# Importer et enregistrer automatiquement les modèles des sous-modules
try:
    from . import unet
except ImportError:
    logger.warning("Module unet non trouvé. Les modèles U-Net ne seront pas disponibles.")
    
try:
    from . import deeplabv3
except ImportError:
    logger.warning("Module deeplabv3 non trouvé. Les modèles DeepLabV3+ ne seront pas disponibles.")

try:
    from . import unet_regression
except ImportError:
    logger.warning("Module unet_regression non trouvé. Les modèles U-Net de régression ne seront pas disponibles.") 