#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemple d'inférence pour ForestGaps.

Ce script montre comment charger un modèle préentraîné et l'utiliser pour faire
des prédictions sur de nouvelles données DSM. Il permet également de visualiser
et sauvegarder les résultats.

Auteur: Arthur VDL
"""

import os
import sys
import argparse
import yaml
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Assurer que le package est dans le PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forestgaps.config import forestgaps.configurationManager
from forestgaps.environment import setup_environment
from forestgaps.models import load_model
from forestgaps.data.normalization import apply_normalization
from forestgaps.utils.visualization import (
    visualize_prediction_overlay,
    visualize_prediction_comparison
)

# ===================================================================================================
# CONFIGURATION ET PARAMÈTRES
# ===================================================================================================

def parse_arguments():
    """
    Analyser les arguments en ligne de commande.
    
    Returns:
        argparse.Namespace: Les arguments analysés.
    """
    parser = argparse.ArgumentParser(
        description='Exemple d\'inférence pour ForestGaps'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Chemin vers le modèle préentraîné (.pt)'
    )
    parser.add_argument(
        '--dsm_dir',
        type=str,
        required=True,
        help='Répertoire contenant les fichiers DSM (.npy)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./inference_output',
        help='Répertoire de sortie pour les résultats'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=5.0,
        help='Seuil de hauteur pour la détection des trouées (en mètres)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Taille des lots pour l\'inférence'
    )
    parser.add_argument(
        '--visualization',
        action='store_true',
        help='Générer des visualisations'
    )
    
    return parser.parse_args()

# ===================================================================================================
# PRÉPARATION DE L'ENVIRONNEMENT
# ===================================================================================================

def setup_workspace(output_dir):
    """
    Configurer les répertoires de travail.
    
    Args:
        output_dir (str): Répertoire de sortie principal.
        
    Returns:
        dict: Dictionnaire des chemins configurés.
    """
    # Créer les sous-répertoires nécessaires
    dirs = {
        "output": output_dir,
        "predictions": os.path.join(output_dir, "predictions"),
        "visualizations": os.path.join(output_dir, "visualizations"),
        "logs": os.path.join(output_dir, "logs")
    }
    
    # Créer les répertoires
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs

def setup_logging(log_dir):
    """
    Configurer la journalisation.
    
    Args:
        log_dir (str): Répertoire pour les fichiers de log.
        
    Returns:
        logging.Logger: Logger configuré.
    """
    logger = logging.getLogger("forestgaps")
    logger.setLevel(logging.INFO)
    
    # Formatter pour les logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Handler pour le fichier
    log_file = os.path.join(log_dir, "inference.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Ajouter les handlers au logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# ===================================================================================================
# FONCTIONS D'INFÉRENCE
# ===================================================================================================

def load_dsm_files(dsm_dir):
    """
    Charger tous les fichiers DSM d'un répertoire.
    
    Args:
        dsm_dir (str): Répertoire contenant les fichiers DSM.
        
    Returns:
        dict: Dictionnaire {nom_fichier: chemin_complet} des fichiers DSM.
    """
    dsm_files = {}
    
    for file in os.listdir(dsm_dir):
        if file.endswith(".npy") and "dsm" in file.lower():
            file_path = os.path.join(dsm_dir, file)
            # Utiliser le nom de fichier sans extension comme clé
            base_name = os.path.splitext(file)[0]
            dsm_files[base_name] = file_path
    
    return dsm_files

def process_batch(model, batch, threshold, device):
    """
    Traiter un lot d'images DSM avec le modèle.
    
    Args:
        model (torch.nn.Module): Modèle préentraîné.
        batch (torch.Tensor): Lot d'images DSM [B, C, H, W].
        threshold (float): Seuil de hauteur pour la détection.
        device (torch.device): Dispositif de traitement (CPU/GPU).
        
    Returns:
        torch.Tensor: Prédictions [B, C, H, W].
    """
    # Déplacer le lot vers le dispositif
    batch = batch.to(device)
    
    # Créer un tenseur de seuil
    threshold_tensor = torch.tensor([threshold], device=device)
    
    # Faire l'inférence
    with torch.no_grad():
        predictions = model(batch, threshold_tensor)
        
    # Appliquer le seuil de probabilité (sigmoid + seuil binaire)
    predictions = torch.sigmoid(predictions) > 0.5
    
    return predictions.float()

def run_inference(model, dsm_files, threshold, batch_size, device, prediction_dir, logger):
    """
    Exécuter l'inférence sur tous les fichiers DSM.
    
    Args:
        model (torch.nn.Module): Modèle préentraîné.
        dsm_files (dict): Dictionnaire des fichiers DSM.
        threshold (float): Seuil de hauteur pour la détection.
        batch_size (int): Taille des lots.
        device (torch.device): Dispositif de traitement.
        prediction_dir (str): Répertoire pour sauvegarder les prédictions.
        logger (logging.Logger): Logger pour les messages.
        
    Returns:
        dict: Résultats de l'inférence {nom_fichier: chemin_prediction}.
    """
    # Mettre le modèle en mode évaluation
    model.eval()
    
    # Dictionnaire pour stocker les résultats
    results = {}
    
    # Traiter les fichiers par lots
    file_names = list(dsm_files.keys())
    num_files = len(file_names)
    
    logger.info(f"Traitement de {num_files} fichiers DSM...")
    
    # Calculer le nombre de lots
    num_batches = (num_files + batch_size - 1) // batch_size
    
    # Traiter les fichiers par lots
    for batch_idx in tqdm(range(num_batches), desc="Traitement des lots"):
        # Calculer les indices de début et fin du lot
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_files)
        batch_files = file_names[start_idx:end_idx]
        
        # Charger les données DSM du lot
        batch_dsm = []
        for file_name in batch_files:
            # Charger le fichier DSM
            dsm_data = np.load(dsm_files[file_name])
            
            # Normaliser les données (min-max)
            dsm_normalized = (dsm_data - dsm_data.min()) / (dsm_data.max() - dsm_data.min() + 1e-8)
            
            # Ajouter la dimension de canal et convertir en tenseur
            dsm_tensor = torch.from_numpy(dsm_normalized).float().unsqueeze(0)  # [C, H, W]
            batch_dsm.append(dsm_tensor)
        
        # Empiler les tenseurs pour former un lot
        batch_tensor = torch.stack(batch_dsm)  # [B, C, H, W]
        
        # Traiter le lot
        batch_predictions = process_batch(model, batch_tensor, threshold, device)
        
        # Sauvegarder les prédictions
        for i, file_name in enumerate(batch_files):
            # Extraire la prédiction
            prediction = batch_predictions[i, 0].cpu().numpy()  # [H, W]
            
            # Définir le chemin de sortie
            output_path = os.path.join(prediction_dir, f"{file_name}_prediction.npy")
            
            # Sauvegarder la prédiction
            np.save(output_path, prediction)
            
            # Stocker le résultat
            results[file_name] = output_path
    
    logger.info(f"{len(results)} prédictions générées et sauvegardées.")
    
    return results

def create_visualizations(dsm_files, prediction_results, threshold, visualization_dir, logger):
    """
    Créer des visualisations des prédictions.
    
    Args:
        dsm_files (dict): Dictionnaire des fichiers DSM.
        prediction_results (dict): Résultats de l'inférence.
        threshold (float): Seuil de hauteur utilisé.
        visualization_dir (str): Répertoire pour les visualisations.
        logger (logging.Logger): Logger pour les messages.
    """
    logger.info("Création des visualisations...")
    
    # Limiter le nombre de visualisations si trop nombreuses
    max_visualizations = 20
    files_to_visualize = list(prediction_results.keys())
    
    if len(files_to_visualize) > max_visualizations:
        logger.info(f"Limitation à {max_visualizations} visualisations sur {len(files_to_visualize)} fichiers.")
        files_to_visualize = files_to_visualize[:max_visualizations]
    
    # Créer les visualisations
    for file_name in tqdm(files_to_visualize, desc="Création des visualisations"):
        # Charger le DSM
        dsm_data = np.load(dsm_files[file_name])
        
        # Charger la prédiction
        prediction = np.load(prediction_results[file_name])
        
        # Créer la visualisation overlay
        overlay_path = os.path.join(visualization_dir, f"{file_name}_overlay.png")
        visualize_prediction_overlay(
            dsm=dsm_data,
            prediction=prediction,
            title=f"Prédiction des trouées (seuil {threshold}m)",
            save_path=overlay_path
        )
        
        # Créer la visualisation comparative
        comparison_path = os.path.join(visualization_dir, f"{file_name}_comparison.png")
        visualize_prediction_comparison(
            dsm=dsm_data,
            prediction=prediction,
            title=f"DSM et prédiction (seuil {threshold}m)",
            save_path=comparison_path
        )
    
    logger.info(f"Visualisations créées et sauvegardées dans: {visualization_dir}")

# ===================================================================================================
# FONCTION PRINCIPALE
# ===================================================================================================

def main():
    """
    Fonction principale pour l'exemple d'inférence.
    """
    # Analyser les arguments
    args = parse_arguments()
    
    # Configurer les répertoires
    dirs = setup_workspace(args.output_dir)
    
    # Configurer la journalisation
    logger = setup_logging(dirs["logs"])
    logger.info("Démarrage de l'exemple d'inférence ForestGaps")
    logger.info(f"Modèle: {args.model_path}")
    logger.info(f"Seuil de hauteur: {args.threshold}m")
    
    try:
        # Configurer l'environnement
        logger.info("Configuration de l'environnement...")
        env = setup_environment()
        logger.info(f"Environnement détecté: {env.name}, GPU: {env.has_gpu}")
        device = env.get_device()
        
        # Charger le modèle
        logger.info("Chargement du modèle...")
        model = load_model(args.model_path)
        model.to(device)
        
        # Charger les fichiers DSM
        logger.info(f"Chargement des fichiers DSM depuis: {args.dsm_dir}")
        dsm_files = load_dsm_files(args.dsm_dir)
        logger.info(f"{len(dsm_files)} fichiers DSM trouvés.")
        
        if len(dsm_files) == 0:
            logger.error(f"Aucun fichier DSM trouvé dans le répertoire: {args.dsm_dir}")
            return
        
        # Exécuter l'inférence
        logger.info("Démarrage de l'inférence...")
        prediction_results = run_inference(
            model=model,
            dsm_files=dsm_files,
            threshold=args.threshold,
            batch_size=args.batch_size,
            device=device,
            prediction_dir=dirs["predictions"],
            logger=logger
        )
        
        # Créer des visualisations si demandé
        if args.visualization:
            create_visualizations(
                dsm_files=dsm_files,
                prediction_results=prediction_results,
                threshold=args.threshold,
                visualization_dir=dirs["visualizations"],
                logger=logger
            )
        
        logger.info("Exemple d'inférence terminé avec succès.")
        
    except Exception as e:
        logger.exception(f"Erreur lors de l'exécution: {str(e)}")
        raise

# ===================================================================================================
# POINT D'ENTRÉE
# ===================================================================================================

if __name__ == "__main__":
    main() 