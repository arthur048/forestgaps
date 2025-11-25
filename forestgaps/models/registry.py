"""
Registre des modèles pour la détection des trouées forestières.

Ce module fournit un registre centralisé pour toutes les architectures
de modèles disponibles dans le package, permettant une création et une
gestion dynamiques des modèles.
"""

import logging
from typing import Dict, Type, Callable, Any, List, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registre global des architectures de modèles disponibles.
    
    Le registre permet d'enregistrer des architectures de modèles et de les
    créer dynamiquement à partir de leur nom, facilitant l'extensibilité
    et la configuration sans modification du code existant.
    
    Example:
        @ModelRegistry.register("unet")
        class UNet(nn.Module):
            def __init__(self, in_channels=1, out_channels=1):
                ...
        
        # Pour créer un modèle:
        model = ModelRegistry.create("unet", in_channels=1, out_channels=1)
    """
    
    _registry: Dict[str, Type[nn.Module]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Décorateur pour enregistrer une classe de modèle dans le registre.
        
        Args:
            name: Nom unique pour identifier le modèle dans le registre
            
        Returns:
            Décorateur qui enregistre la classe de modèle
            
        Example:
            @ModelRegistry.register("unet_film")
            class UNetWithFiLM(nn.Module):
                ...
        """
        def decorator(model_class: Type[nn.Module]) -> Type[nn.Module]:
            if name in cls._registry:
                logger.warning(f"Le modèle '{name}' est déjà enregistré. Il sera remplacé.")
            
            cls._registry[name] = model_class
            logger.debug(f"Modèle '{name}' enregistré avec succès")
            return model_class
        
        return decorator
    
    @classmethod
    def create(cls, model_type: str, **kwargs: Any) -> nn.Module:
        """
        Crée une instance de modèle à partir du registre.
        
        Args:
            model_type: Nom du modèle à créer
            **kwargs: Arguments à passer au constructeur du modèle
            
        Returns:
            Instance du modèle demandé
            
        Raises:
            ValueError: Si le modèle demandé n'est pas dans le registre
            
        Example:
            model = ModelRegistry.create("unet_cbam", in_channels=1, out_channels=1)
        """
        if model_type not in cls._registry:
            available_models = list(cls._registry.keys())
            raise ValueError(f"Modèle '{model_type}' non trouvé dans le registre. "
                             f"Options disponibles: {available_models}")
        
        model_class = cls._registry[model_type]
        logger.info(f"Création du modèle '{model_type}'")
        return model_class(**kwargs)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """
        Retourne la liste des modèles disponibles dans le registre.
        
        Returns:
            Liste des noms de modèles enregistrés
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_model_class(cls, model_type: str) -> Optional[Type[nn.Module]]:
        """
        Retourne la classe du modèle sans l'instancier.
        
        Args:
            model_type: Nom du modèle à récupérer
            
        Returns:
            Classe du modèle ou None si non trouvée
        """
        return cls._registry.get(model_type)


def get_model_from_config(config: dict) -> nn.Module:
    """
    Crée un modèle à partir d'une configuration.
    
    Args:
        config: Dictionnaire de configuration contenant:
            - model_type: Type de modèle à créer
            - model_params: Paramètres à passer au constructeur du modèle
            
    Returns:
        Instance du modèle demandé
    
    Example:
        config = {
            "model_type": "unet",
            "model_params": {
                "in_channels": 1,
                "out_channels": 1,
                "init_features": 32
            }
        }
        model = get_model_from_config(config)
    """
    model_type = config.get("model_type")
    model_params = config.get("model_params", {})
    
    if not model_type:
        raise ValueError("La configuration doit contenir un 'model_type'")
    
    return ModelRegistry.create(model_type, **model_params) 