"""
Fonctionnalités d'export au format TorchScript pour les modèles de détection des trouées forestières.

Ce module fournit des fonctions pour exporter les modèles PyTorch au format TorchScript,
ce qui permet leur déploiement dans des environnements sans dépendance à Python.
"""

import os
import logging
import json
from typing import Dict, Any, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.jit as jit

from ..base import ForestGapModel, ThresholdConditionedModel


def export_to_torchscript(
    model: ForestGapModel,
    save_path: str,
    input_shape: Tuple[int, ...],
    method: str = "trace",
    optimize: bool = True,
    check_model: bool = True,
    verbose: bool = False
) -> str:
    """
    Exporte un modèle PyTorch au format TorchScript.
    
    Args:
        model: Modèle PyTorch à exporter
        save_path: Chemin où sauvegarder le modèle TorchScript
        input_shape: Forme de l'entrée (B, C, H, W)
        method: Méthode d'export ('trace' ou 'script')
        optimize: Si True, optimise le modèle TorchScript
        check_model: Si True, vérifie la validité du modèle TorchScript
        verbose: Si True, affiche des informations détaillées
        
    Returns:
        Chemin du modèle TorchScript exporté
        
    Raises:
        ValueError: Si la méthode d'export n'est pas valide
        RuntimeError: Si l'export échoue
    """
    # Vérifier l'extension du fichier
    if not save_path.endswith(".pt") and not save_path.endswith(".pth"):
        save_path += ".pt"
        
    # Vérifier la méthode d'export
    if method not in ["trace", "script"]:
        raise ValueError("La méthode d'export doit être 'trace' ou 'script'")
    
    # Mettre le modèle en mode évaluation
    model.eval()
    
    try:
        if verbose:
            logging.info(f"Exportation du modèle vers {save_path} avec la méthode '{method}'")
            
        # Créer des entrées factices
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_shape, device=device)
        
        # Exporter le modèle selon la méthode choisie
        if method == "trace":
            # Ajouter un seuil factice si nécessaire
            if isinstance(model, ThresholdConditionedModel):
                dummy_threshold = torch.tensor([[0.5]], device=device)
                script_model = jit.trace(model, (dummy_input, dummy_threshold))
            else:
                script_model = jit.trace(model, dummy_input)
        else:  # method == "script"
            script_model = jit.script(model)
        
        # Optimiser le modèle si demandé
        if optimize:
            script_model = jit.optimize_for_inference(script_model)
            
        # Sauvegarder le modèle
        jit.save(script_model, save_path)
        
        # Vérifier le modèle si demandé
        if check_model:
            # Charger le modèle pour vérifier qu'il peut être chargé correctement
            loaded_model = jit.load(save_path)
            
            # Vérifier que le modèle peut être exécuté
            if isinstance(model, ThresholdConditionedModel):
                loaded_model(dummy_input, dummy_threshold)
            else:
                loaded_model(dummy_input)
                
            if verbose:
                logging.info("Modèle TorchScript vérifié avec succès")
        
        return save_path
        
    except Exception as e:
        logging.error(f"Erreur lors de l'exportation du modèle TorchScript: {str(e)}")
        raise RuntimeError(f"Échec de l'exportation TorchScript: {str(e)}")


def create_scriptable_wrapper(model: ThresholdConditionedModel) -> jit.ScriptModule:
    """
    Crée un wrapper scriptable pour un modèle conditionné par un seuil.
    
    Cette fonction est utile pour les modèles qui ne peuvent pas être directement
    scriptés en raison de constructions Python complexes.
    
    Args:
        model: Modèle conditionné par un seuil
        
    Returns:
        Module TorchScript wrapper
    """
    class ScriptableWrapper(jit.ScriptModule):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        @jit.script_method
        def forward(self, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
            return self.model(x, threshold)
    
    return ScriptableWrapper(model)


def compare_torchscript_outputs(
    model: ForestGapModel,
    script_path: str,
    input_shape: Tuple[int, ...],
    rtol: float = 1e-3,
    atol: float = 1e-5,
    num_samples: int = 5,
    verbose: bool = False
) -> bool:
    """
    Compare les sorties du modèle PyTorch et du modèle TorchScript pour vérifier la cohérence.
    
    Args:
        model: Modèle PyTorch original
        script_path: Chemin du modèle TorchScript
        input_shape: Forme de l'entrée (B, C, H, W)
        rtol: Tolérance relative pour la comparaison
        atol: Tolérance absolue pour la comparaison
        num_samples: Nombre d'échantillons à tester
        verbose: Si True, affiche des informations détaillées
        
    Returns:
        True si les sorties sont cohérentes, False sinon
    """
    # Mettre le modèle en mode évaluation
    model.eval()
    device = next(model.parameters()).device
    
    # Charger le modèle TorchScript
    script_model = jit.load(script_path)
    script_model.eval()
    
    # Tester plusieurs échantillons
    for i in range(num_samples):
        # Créer une entrée aléatoire
        dummy_input = torch.randn(*input_shape, device=device)
        
        # Préparer les entrées et exécuter les modèles
        if isinstance(model, ThresholdConditionedModel):
            dummy_threshold = torch.tensor([[0.5]], device=device)
            
            # Inférence PyTorch
            with torch.no_grad():
                torch_output = model(dummy_input, dummy_threshold)
            
            # Inférence TorchScript
            with torch.no_grad():
                script_output = script_model(dummy_input, dummy_threshold)
        else:
            # Inférence PyTorch
            with torch.no_grad():
                torch_output = model(dummy_input)
            
            # Inférence TorchScript
            with torch.no_grad():
                script_output = script_model(dummy_input)
        
        # Comparer les sorties
        is_close = torch.allclose(torch_output, script_output, rtol=rtol, atol=atol)
        
        if verbose:
            if is_close:
                logging.info(f"Échantillon {i+1}/{num_samples}: Sorties cohérentes")
            else:
                logging.warning(f"Échantillon {i+1}/{num_samples}: Sorties incohérentes")
                logging.warning(f"  Différence max: {torch.max(torch.abs(torch_output - script_output))}")
                logging.warning(f"  Différence moyenne: {torch.mean(torch.abs(torch_output - script_output))}")
        
        if not is_close:
            return False
    
    return True 