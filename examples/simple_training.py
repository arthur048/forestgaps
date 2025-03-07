#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemple d'entraînement simple pour ForestGaps-DL.

Ce script montre comment configurer et entraîner un modèle U-Net
pour la segmentation de trouées forestières avec un ensemble minimal
de paramètres et options.

Auteur: Arthur VDL
"""

import os
import sys
import argparse
import yaml
import logging
import torch
from pathlib import Path

# Assurer que le package est dans le PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forestgaps_dl.config import ConfigurationManager
from forestgaps_dl.environment import setup_environment
from forestgaps_dl.data.loaders import create_data_loaders
from forestgaps_dl.models import create_model
from forestgaps_dl.training import Trainer
from forestgaps_dl.training.callbacks import (
    ModelCheckpointCallback,
    EarlyStoppingCallback,
    TensorBoardCallback
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
        description='Exemple d\'entraînement simple pour ForestGaps-DL'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Répertoire contenant les données prétraitées'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Répertoire de sortie pour les résultats'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['unet', 'unet_film', 'unet_cbam'],
        default='unet',
        help='Type de modèle à utiliser'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Taille des lots pour l\'entraînement'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Nombre d\'époques pour l\'entraînement'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Taux d\'apprentissage initial'
    )
    
    return parser.parse_args()

# ===================================================================================================
# PRÉPARATION DE L'ENVIRONNEMENT
# ===================================================================================================

def setup_workspace(config):
    """
    Configurer les répertoires de travail.
    
    Args:
        config (ConfigurationManager): Configuration du workspace.
        
    Returns:
        dict: Dictionnaire des chemins configurés.
    """
    # Extraire les répertoires de base
    output_dir = config.get("output_dir")
    
    # Créer les sous-répertoires nécessaires
    dirs = {
        "output": output_dir,
        "models": os.path.join(output_dir, "models"),
        "logs": os.path.join(output_dir, "logs"),
        "tensorboard": os.path.join(output_dir, "tensorboard")
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
    logger = logging.getLogger("forestgaps_dl")
    logger.setLevel(logging.INFO)
    
    # Formatter pour les logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Handler pour le fichier
    log_file = os.path.join(log_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Ajouter les handlers au logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# ===================================================================================================
# FONCTION PRINCIPALE
# ===================================================================================================

def main():
    """
    Fonction principale pour l'exemple d'entraînement.
    """
    # Analyser les arguments
    args = parse_arguments()
    
    # Construire une configuration minimale
    config_data = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "data": {
            "processed_dir": os.path.join(args.data_dir, "processed"),
            "tiles_dir": os.path.join(args.data_dir, "tiles"),
            "gap_thresholds": [2.0, 5.0, 10.0, 15.0],
            "normalization_method": "min-max"
        },
        "model": {
            "type": args.model_type,
            "params": {
                "in_channels": 1,
                "dropout_rate": 0.2
            }
        },
        "training": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-5,
            "gradient_clipping": 1.0,
            "early_stopping_patience": 10
        }
    }
    
    # Créer le gestionnaire de configuration
    config = ConfigurationManager(config_data)
    
    # Configurer les répertoires
    dirs = setup_workspace(config)
    
    # Configurer la journalisation
    logger = setup_logging(dirs["logs"])
    logger.info("Démarrage de l'exemple d'entraînement ForestGaps-DL")
    logger.info(f"Type de modèle: {args.model_type}")
    logger.info(f"Taille de batch: {args.batch_size}")
    logger.info(f"Nombre d'époques: {args.epochs}")
    
    # Sauvegarder la configuration
    config_path = os.path.join(dirs["output"], "config.yaml")
    config.save_config(config_path)
    logger.info(f"Configuration sauvegardée dans: {config_path}")
    
    try:
        # Configurer l'environnement
        logger.info("Configuration de l'environnement...")
        env = setup_environment()
        logger.info(f"Environnement détecté: {env.name}, GPU: {env.has_gpu}")
        device = env.get_device()
        
        # Créer les DataLoaders
        logger.info("Chargement des données...")
        data_loaders = create_data_loaders(
            config=config,
            batch_size=args.batch_size,
            num_workers=4 if env.name == "local" else 2
        )
        
        # Créer le modèle
        logger.info(f"Création du modèle: {args.model_type}...")
        model = create_model(
            model_type=args.model_type,
            **config.get("model.params", {})
        )
        
        # Configurer les callbacks
        logger.info("Configuration des callbacks...")
        callbacks = [
            ModelCheckpointCallback(
                save_dir=dirs["models"],
                save_best_only=True,
                metric_name="val_iou",
                mode="max"
            ),
            EarlyStoppingCallback(
                patience=10,
                metric_name="val_loss",
                mode="min"
            ),
            TensorBoardCallback(
                log_dir=dirs["tensorboard"]
            )
        ]
        
        # Créer le Trainer
        logger.info("Initialisation du Trainer...")
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=data_loaders['train'],
            val_loader=data_loaders['val'],
            test_loader=data_loaders.get('test'),
            callbacks=callbacks,
            device=device
        )
        
        # Entraîner le modèle
        logger.info("Démarrage de l'entraînement...")
        results = trainer.train(
            epochs=args.epochs,
            gradient_clipping=config.get("training.gradient_clipping", 1.0)
        )
        
        # Sauvegarder les résultats
        logger.info("Entraînement terminé. Sauvegarde des résultats...")
        trainer.save_training_summary(os.path.join(dirs["output"], "training_summary.json"))
        
        # Évaluer le modèle sur les données de test
        logger.info("Évaluation du modèle sur les données de test...")
        test_results = trainer.evaluate(data_loaders['test'])
        
        # Afficher les résultats de l'évaluation
        logger.info("Résultats de l'évaluation:")
        for threshold, metrics in test_results.get("by_threshold", {}).items():
            logger.info(f"Seuil {threshold}m:")
            logger.info(f"  IoU: {metrics.get('iou', 0):.4f}")
            logger.info(f"  F1: {metrics.get('f1', 0):.4f}")
            logger.info(f"  Précision: {metrics.get('precision', 0):.4f}")
            logger.info(f"  Rappel: {metrics.get('recall', 0):.4f}")
        
        logger.info(f"IoU moyen: {test_results.get('mean_iou', 0):.4f}")
        logger.info(f"F1 moyen: {test_results.get('mean_f1', 0):.4f}")
        
        # Sauvegarder les résultats de l'évaluation
        eval_path = os.path.join(dirs["output"], "evaluation_results.yaml")
        with open(eval_path, 'w') as f:
            yaml.dump(test_results, f)
        logger.info(f"Résultats de l'évaluation sauvegardés dans: {eval_path}")
        
        logger.info("Exemple d'entraînement terminé avec succès.")
        
    except Exception as e:
        logger.exception(f"Erreur lors de l'exécution: {str(e)}")
        raise

# ===================================================================================================
# POINT D'ENTRÉE
# ===================================================================================================

if __name__ == "__main__":
    main() 