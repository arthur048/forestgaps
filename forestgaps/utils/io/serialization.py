# Fonctions de sérialisation/désérialisation
"""
Fonctions de sérialisation/désérialisation pour ForestGaps.

Ce module fournit des fonctions pour sérialiser et désérialiser des objets
utilisés dans le workflow ForestGaps, notamment les modèles, les configurations
et les résultats.
"""

import os
import json
import pickle
import yaml
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

from forestgaps.utils.errors import DataProcessingError


def save_json(data: Dict, file_path: str, indent: int = 4) -> None:
    """
    Sauvegarde des données au format JSON.
    
    Args:
        data (dict): Données à sauvegarder.
        file_path (str): Chemin où sauvegarder le fichier.
        indent (int): Indentation pour le formatage du JSON.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors de la sauvegarde.
    """
    try:
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convertir les types numpy en types Python natifs
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Convertir les données
        converted_data = convert_numpy(data)
        
        # Écrire le fichier JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=indent, ensure_ascii=False)
    except Exception as e:
        raise DataProcessingError(f"Erreur lors de la sauvegarde du fichier JSON {file_path}: {str(e)}")


def load_json(file_path: str) -> Dict:
    """
    Charge des données depuis un fichier JSON.
    
    Args:
        file_path (str): Chemin vers le fichier JSON.
        
    Returns:
        dict: Données chargées.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors du chargement.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise DataProcessingError(f"Erreur lors du chargement du fichier JSON {file_path}: {str(e)}")


def save_yaml(data: Dict, file_path: str) -> None:
    """
    Sauvegarde des données au format YAML.
    
    Args:
        data (dict): Données à sauvegarder.
        file_path (str): Chemin où sauvegarder le fichier.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors de la sauvegarde.
    """
    try:
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convertir les types numpy en types Python natifs
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Convertir les données
        converted_data = convert_numpy(data)
        
        # Écrire le fichier YAML
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(converted_data, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise DataProcessingError(f"Erreur lors de la sauvegarde du fichier YAML {file_path}: {str(e)}")


def load_yaml(file_path: str) -> Dict:
    """
    Charge des données depuis un fichier YAML.
    
    Args:
        file_path (str): Chemin vers le fichier YAML.
        
    Returns:
        dict: Données chargées.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors du chargement.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise DataProcessingError(f"Erreur lors du chargement du fichier YAML {file_path}: {str(e)}")


def save_pickle(data: Any, file_path: str) -> None:
    """
    Sauvegarde des données au format pickle.
    
    Args:
        data: Données à sauvegarder.
        file_path (str): Chemin où sauvegarder le fichier.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors de la sauvegarde.
    """
    try:
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Écrire le fichier pickle
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise DataProcessingError(f"Erreur lors de la sauvegarde du fichier pickle {file_path}: {str(e)}")


def load_pickle(file_path: str) -> Any:
    """
    Charge des données depuis un fichier pickle.
    
    Args:
        file_path (str): Chemin vers le fichier pickle.
        
    Returns:
        Any: Données chargées.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors du chargement.
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise DataProcessingError(f"Erreur lors du chargement du fichier pickle {file_path}: {str(e)}")


def save_model(model: torch.nn.Module, file_path: str, optimizer: Optional[torch.optim.Optimizer] = None,
              epoch: Optional[int] = None, metrics: Optional[Dict] = None) -> None:
    """
    Sauvegarde un modèle PyTorch avec des informations supplémentaires.
    
    Args:
        model (torch.nn.Module): Modèle à sauvegarder.
        file_path (str): Chemin où sauvegarder le modèle.
        optimizer (torch.optim.Optimizer, optional): Optimiseur à sauvegarder.
        epoch (int, optional): Époque actuelle.
        metrics (dict, optional): Métriques à sauvegarder.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors de la sauvegarde.
    """
    try:
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Préparer les données à sauvegarder
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        # Sauvegarder le checkpoint
        torch.save(checkpoint, file_path)
    except Exception as e:
        raise DataProcessingError(f"Erreur lors de la sauvegarde du modèle {file_path}: {str(e)}")


def load_model(file_path: str, model: Optional[torch.nn.Module] = None,
              optimizer: Optional[torch.optim.Optimizer] = None,
              device: Optional[torch.device] = None) -> Dict:
    """
    Charge un modèle PyTorch avec des informations supplémentaires.
    
    Args:
        file_path (str): Chemin vers le fichier du modèle.
        model (torch.nn.Module, optional): Modèle à charger.
        optimizer (torch.optim.Optimizer, optional): Optimiseur à charger.
        device (torch.device, optional): Périphérique où charger le modèle.
        
    Returns:
        dict: Checkpoint chargé.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors du chargement.
    """
    try:
        # Déterminer le périphérique
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Charger le checkpoint
        checkpoint = torch.load(file_path, map_location=device)
        
        # Charger l'état du modèle si un modèle est fourni
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
        
        # Charger l'état de l'optimiseur si un optimiseur est fourni
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    except Exception as e:
        raise DataProcessingError(f"Erreur lors du chargement du modèle {file_path}: {str(e)}")


def save_numpy(data: np.ndarray, file_path: str) -> None:
    """
    Sauvegarde un tableau NumPy.
    
    Args:
        data (numpy.ndarray): Tableau à sauvegarder.
        file_path (str): Chemin où sauvegarder le tableau.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors de la sauvegarde.
    """
    try:
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Sauvegarder le tableau
        np.save(file_path, data)
    except Exception as e:
        raise DataProcessingError(f"Erreur lors de la sauvegarde du tableau NumPy {file_path}: {str(e)}")


def load_numpy(file_path: str) -> np.ndarray:
    """
    Charge un tableau NumPy.
    
    Args:
        file_path (str): Chemin vers le fichier du tableau.
        
    Returns:
        numpy.ndarray: Tableau chargé.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors du chargement.
    """
    try:
        return np.load(file_path)
    except Exception as e:
        raise DataProcessingError(f"Erreur lors du chargement du tableau NumPy {file_path}: {str(e)}")


def export_model_to_onnx(model: torch.nn.Module, file_path: str, input_shape: Tuple[int, ...],
                        threshold_shape: Tuple[int, ...], dynamic_axes: Optional[Dict] = None) -> None:
    """
    Exporte un modèle PyTorch au format ONNX.
    
    Args:
        model (torch.nn.Module): Modèle à exporter.
        file_path (str): Chemin où sauvegarder le modèle ONNX.
        input_shape (tuple): Forme de l'entrée du modèle.
        threshold_shape (tuple): Forme du tenseur de seuil.
        dynamic_axes (dict, optional): Axes dynamiques pour l'exportation.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors de l'exportation.
    """
    try:
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Mettre le modèle en mode évaluation
        model.eval()
        
        # Créer des tenseurs d'exemple
        device = next(model.parameters()).device
        dummy_input = (
            torch.zeros(input_shape, device=device),
            torch.zeros(threshold_shape, device=device)
        )
        
        # Définir les axes dynamiques par défaut si non fournis
        if dynamic_axes is None:
            dynamic_axes = {
                'dsm': {0: 'batch_size'},
                'threshold': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Exporter le modèle
        torch.onnx.export(
            model,
            dummy_input,
            file_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['dsm', 'threshold'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
    except Exception as e:
        raise DataProcessingError(f"Erreur lors de l'exportation du modèle au format ONNX {file_path}: {str(e)}")


def export_model_to_torchscript(model: torch.nn.Module, file_path: str, input_shape: Tuple[int, ...],
                               threshold_shape: Tuple[int, ...]) -> None:
    """
    Exporte un modèle PyTorch au format TorchScript.
    
    Args:
        model (torch.nn.Module): Modèle à exporter.
        file_path (str): Chemin où sauvegarder le modèle TorchScript.
        input_shape (tuple): Forme de l'entrée du modèle.
        threshold_shape (tuple): Forme du tenseur de seuil.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors de l'exportation.
    """
    try:
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Mettre le modèle en mode évaluation
        model.eval()
        
        # Créer des tenseurs d'exemple
        device = next(model.parameters()).device
        dummy_input = (
            torch.zeros(input_shape, device=device),
            torch.zeros(threshold_shape, device=device)
        )
        
        # Tracer le modèle
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Sauvegarder le modèle
        torch.jit.save(traced_model, file_path)
    except Exception as e:
        raise DataProcessingError(f"Erreur lors de l'exportation du modèle au format TorchScript {file_path}: {str(e)}")
