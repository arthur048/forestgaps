"""
Fonctionnalités d'export au format ONNX pour les modèles de détection des trouées forestières.

Ce module fournit des fonctions pour exporter les modèles PyTorch au format ONNX,
optimiser les modèles ONNX et vérifier leur validité.
"""

import os
import logging
import json
from typing import Dict, Any, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import numpy as np

# Vérifier si onnx est disponible
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX ou ONNXRuntime non disponible. L'export ONNX ne sera pas possible.")

# Vérifier si onnx-simplifier est disponible
try:
    import onnxsim
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False
    logging.warning("ONNX-Simplifier non disponible. L'optimisation des modèles ONNX sera limitée.")

from ..base import ForestGapModel, ThresholdConditionedModel


def export_to_onnx(
    model: ForestGapModel,
    save_path: str,
    input_shape: Tuple[int, ...],
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 11,
    optimize: bool = True,
    check_model: bool = True,
    export_params: bool = True,
    verbose: bool = False
) -> str:
    """
    Exporte un modèle PyTorch au format ONNX.
    
    Args:
        model: Modèle PyTorch à exporter
        save_path: Chemin où sauvegarder le modèle ONNX
        input_shape: Forme de l'entrée (B, C, H, W)
        input_names: Noms des entrées du modèle
        output_names: Noms des sorties du modèle
        dynamic_axes: Axes dynamiques pour l'inférence avec des tailles variables
        opset_version: Version de l'opset ONNX à utiliser
        optimize: Si True, optimise le modèle ONNX après l'export
        check_model: Si True, vérifie la validité du modèle ONNX
        export_params: Si True, exporte les paramètres du modèle
        verbose: Si True, affiche des informations détaillées
        
    Returns:
        Chemin du modèle ONNX exporté
        
    Raises:
        ImportError: Si ONNX n'est pas disponible
        RuntimeError: Si l'export échoue
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX ou ONNXRuntime non disponible. Installez-les avec 'pip install onnx onnxruntime'.")
    
    # Vérifier l'extension du fichier
    if not save_path.endswith(".onnx"):
        save_path += ".onnx"
        
    # Configurer les noms d'entrée et de sortie par défaut
    if input_names is None:
        if isinstance(model, ThresholdConditionedModel):
            input_names = ["input", "threshold"]
        else:
            input_names = ["input"]
            
    if output_names is None:
        output_names = ["output"]
        
    # Configurer les axes dynamiques par défaut
    if dynamic_axes is None:
        if isinstance(model, ThresholdConditionedModel):
            dynamic_axes = {
                "input": {0: "batch_size"},
                "threshold": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        else:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
    
    # Mettre le modèle en mode évaluation
    model.eval()
    
    # Créer des entrées factices
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Ajouter un seuil factice si nécessaire
    if isinstance(model, ThresholdConditionedModel):
        dummy_threshold = torch.tensor([[0.5]], device=device)
        dummy_inputs = (dummy_input, dummy_threshold)
    else:
        dummy_inputs = dummy_input
    
    # Exporter le modèle
    try:
        if verbose:
            logging.info(f"Exportation du modèle vers {save_path}")
            
        torch.onnx.export(
            model,
            dummy_inputs,
            save_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            export_params=export_params,
            do_constant_folding=True,
            verbose=verbose
        )
        
        if check_model:
            # Vérifier le modèle ONNX
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            if verbose:
                logging.info("Modèle ONNX vérifié avec succès")
        
        if optimize and ONNXSIM_AVAILABLE:
            # Optimiser le modèle ONNX
            optimize_onnx_model(save_path, save_path, check_model=check_model, verbose=verbose)
            
        return save_path
        
    except Exception as e:
        logging.error(f"Erreur lors de l'exportation du modèle ONNX: {str(e)}")
        raise RuntimeError(f"Échec de l'exportation ONNX: {str(e)}")


def optimize_onnx_model(
    input_path: str,
    output_path: Optional[str] = None,
    check_model: bool = True,
    verbose: bool = False
) -> str:
    """
    Optimise un modèle ONNX en utilisant onnx-simplifier.
    
    Args:
        input_path: Chemin du modèle ONNX à optimiser
        output_path: Chemin où sauvegarder le modèle optimisé (si None, écrase l'original)
        check_model: Si True, vérifie la validité du modèle optimisé
        verbose: Si True, affiche des informations détaillées
        
    Returns:
        Chemin du modèle ONNX optimisé
        
    Raises:
        ImportError: Si onnx-simplifier n'est pas disponible
        RuntimeError: Si l'optimisation échoue
    """
    if not ONNXSIM_AVAILABLE:
        raise ImportError("ONNX-Simplifier non disponible. Installez-le avec 'pip install onnxsim'.")
    
    if output_path is None:
        output_path = input_path
    
    try:
        if verbose:
            logging.info(f"Optimisation du modèle ONNX: {input_path} -> {output_path}")
            
        # Charger le modèle
        onnx_model = onnx.load(input_path)
        
        # Optimiser le modèle
        model_optimized, check_ok = onnxsim.simplify(onnx_model)
        
        if not check_ok:
            logging.warning("La vérification du modèle optimisé a échoué, mais l'optimisation continue")
        
        # Sauvegarder le modèle optimisé
        onnx.save(model_optimized, output_path)
        
        if check_model:
            # Vérifier le modèle optimisé
            onnx.checker.check_model(model_optimized)
            if verbose:
                logging.info("Modèle ONNX optimisé vérifié avec succès")
        
        return output_path
        
    except Exception as e:
        logging.error(f"Erreur lors de l'optimisation du modèle ONNX: {str(e)}")
        raise RuntimeError(f"Échec de l'optimisation ONNX: {str(e)}")


def compare_onnx_outputs(
    model: ForestGapModel,
    onnx_path: str,
    input_shape: Tuple[int, ...],
    rtol: float = 1e-3,
    atol: float = 1e-5,
    num_samples: int = 5,
    verbose: bool = False
) -> bool:
    """
    Compare les sorties du modèle PyTorch et du modèle ONNX pour vérifier la cohérence.
    
    Args:
        model: Modèle PyTorch original
        onnx_path: Chemin du modèle ONNX
        input_shape: Forme de l'entrée (B, C, H, W)
        rtol: Tolérance relative pour la comparaison
        atol: Tolérance absolue pour la comparaison
        num_samples: Nombre d'échantillons à tester
        verbose: Si True, affiche des informations détaillées
        
    Returns:
        True si les sorties sont cohérentes, False sinon
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX ou ONNXRuntime non disponible. Installez-les avec 'pip install onnx onnxruntime'.")
    
    # Mettre le modèle en mode évaluation
    model.eval()
    device = next(model.parameters()).device
    
    # Créer une session ONNX
    ort_session = ort.InferenceSession(onnx_path)
    
    # Tester plusieurs échantillons
    for i in range(num_samples):
        # Créer une entrée aléatoire
        dummy_input = torch.randn(*input_shape, device=device)
        
        # Préparer les entrées pour ONNX
        if isinstance(model, ThresholdConditionedModel):
            dummy_threshold = torch.tensor([[0.5]], device=device)
            
            # Inférence PyTorch
            with torch.no_grad():
                torch_output = model(dummy_input, dummy_threshold).cpu().numpy()
            
            # Inférence ONNX
            ort_inputs = {
                "input": dummy_input.cpu().numpy(),
                "threshold": dummy_threshold.cpu().numpy()
            }
        else:
            # Inférence PyTorch
            with torch.no_grad():
                torch_output = model(dummy_input).cpu().numpy()
            
            # Inférence ONNX
            ort_inputs = {"input": dummy_input.cpu().numpy()}
        
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        # Comparer les sorties
        is_close = np.allclose(torch_output, ort_output, rtol=rtol, atol=atol)
        
        if verbose:
            if is_close:
                logging.info(f"Échantillon {i+1}/{num_samples}: Sorties cohérentes")
            else:
                logging.warning(f"Échantillon {i+1}/{num_samples}: Sorties incohérentes")
                logging.warning(f"  Différence max: {np.max(np.abs(torch_output - ort_output))}")
                logging.warning(f"  Différence moyenne: {np.mean(np.abs(torch_output - ort_output))}")
        
        if not is_close:
            return False
    
    return True 