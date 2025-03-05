"""
Utilitaires pour l'export des modèles de détection des trouées forestières.

Ce module fournit des fonctions utilitaires pour l'export des modèles,
comme le traçage, la sauvegarde des métadonnées, etc.
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, List, Tuple, Optional, Union

import torch
import torch.nn as nn

from ..base import ForestGapModel


def trace_model(
    model: ForestGapModel,
    input_shape: Tuple[int, ...],
    example_inputs: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
    check_trace: bool = True,
    strict: bool = True,
    verbose: bool = False
) -> torch.jit.ScriptModule:
    """
    Trace un modèle PyTorch pour l'export.
    
    Args:
        model: Modèle PyTorch à tracer
        input_shape: Forme de l'entrée (B, C, H, W)
        example_inputs: Entrées d'exemple pour le traçage (si None, générées automatiquement)
        check_trace: Si True, vérifie la validité du traçage
        strict: Si True, utilise un traçage strict
        verbose: Si True, affiche des informations détaillées
        
    Returns:
        Module TorchScript tracé
    """
    # Mettre le modèle en mode évaluation
    model.eval()
    
    # Créer des entrées d'exemple si non fournies
    if example_inputs is None:
        device = next(model.parameters()).device
        example_inputs = torch.randn(*input_shape, device=device)
    
    # Tracer le modèle
    try:
        if verbose:
            logging.info("Traçage du modèle...")
            
        traced_model = torch.jit.trace(model, example_inputs, check_trace=check_trace, strict=strict)
        
        if verbose:
            logging.info("Modèle tracé avec succès")
            
        return traced_model
        
    except Exception as e:
        logging.error(f"Erreur lors du traçage du modèle: {str(e)}")
        raise RuntimeError(f"Échec du traçage: {str(e)}")


def save_model_info(
    model: ForestGapModel,
    save_path: str,
    input_shape: Tuple[int, ...],
    additional_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Sauvegarde les informations sur un modèle dans un fichier JSON.
    
    Args:
        model: Modèle PyTorch
        save_path: Chemin où sauvegarder les informations
        input_shape: Forme de l'entrée (B, C, H, W)
        additional_info: Informations supplémentaires à inclure
        
    Returns:
        Chemin du fichier JSON
    """
    # Vérifier l'extension du fichier
    if not save_path.endswith(".json"):
        save_path += ".json"
    
    # Collecter les informations de base sur le modèle
    model_info = {
        "model_type": model.__class__.__name__,
        "in_channels": model.in_channels,
        "out_channels": model.out_channels,
        "input_shape": list(input_shape),
        "parameters": model.get_num_parameters(),
        "complexity": model.get_complexity(),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    
    # Ajouter des informations supplémentaires si fournies
    if additional_info is not None:
        model_info.update(additional_info)
    
    # Sauvegarder les informations dans un fichier JSON
    try:
        with open(save_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        return save_path
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde des informations du modèle: {str(e)}")
        raise RuntimeError(f"Échec de la sauvegarde des informations: {str(e)}")


def get_model_size(model: nn.Module) -> Dict[str, Union[int, float]]:
    """
    Calcule la taille d'un modèle en mémoire.
    
    Args:
        model: Modèle PyTorch
        
    Returns:
        Dictionnaire contenant la taille du modèle en octets, Ko, Mo
    """
    # Calculer la taille totale des paramètres
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    # Calculer la taille des tampons (buffers)
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    # Taille totale
    total_size = param_size + buffer_size
    
    return {
        "size_bytes": total_size,
        "size_kb": total_size / 1024,
        "size_mb": total_size / (1024 * 1024),
        "parameters_bytes": param_size,
        "buffers_bytes": buffer_size
    }


def create_model_summary(
    model: ForestGapModel,
    input_shape: Tuple[int, ...],
    include_layer_details: bool = False
) -> Dict[str, Any]:
    """
    Crée un résumé détaillé d'un modèle.
    
    Args:
        model: Modèle PyTorch
        input_shape: Forme de l'entrée (B, C, H, W)
        include_layer_details: Si True, inclut les détails de chaque couche
        
    Returns:
        Dictionnaire contenant le résumé du modèle
    """
    # Informations de base
    summary = {
        "model_type": model.__class__.__name__,
        "in_channels": model.in_channels,
        "out_channels": model.out_channels,
        "input_shape": list(input_shape),
        "parameters": model.get_num_parameters(),
        "size": get_model_size(model),
        "complexity": model.get_complexity()
    }
    
    # Ajouter les détails des couches si demandé
    if include_layer_details:
        layers_info = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.BatchNorm2d)):
                layer_info = {
                    "name": name,
                    "type": module.__class__.__name__,
                }
                
                # Ajouter des informations spécifiques selon le type de couche
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    layer_info.update({
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels,
                        "kernel_size": module.kernel_size,
                        "stride": module.stride,
                        "padding": module.padding
                    })
                elif isinstance(module, nn.Linear):
                    layer_info.update({
                        "in_features": module.in_features,
                        "out_features": module.out_features
                    })
                
                # Calculer le nombre de paramètres
                num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                layer_info["parameters"] = num_params
                
                layers_info.append(layer_info)
        
        summary["layers"] = layers_info
    
    return summary 