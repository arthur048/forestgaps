#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exemple d'utilisation du module de benchmarking pour comparer différents modèles.

Ce script montre comment utiliser le module de benchmarking pour comparer
les performances de différents modèles (U-Net et DeepLabV3+) sur le jeu
de données de trouées forestières.
"""

import os
import argparse
import logging
from pathlib import Path

from forestgaps.config import load_config_from_file, load_default_config
from forestgaps.environment import setup_environment
from forestgaps.data.loaders import create_data_loaders
from forestgaps.benchmarking import ModelComparison


# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("benchmark_example")


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Benchmarking des modèles ForestGaps")
    
    parser.add_argument("--config", type=str, default=None,
                        help="Chemin vers le fichier de configuration.")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Répertoire de sortie pour les résultats.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Nombre d'époques pour l'entraînement.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Taille du batch pour l'entraînement.")
    parser.add_argument("--quick-mode", action="store_true",
                        help="Mode rapide avec un sous-ensemble des données.")
    
    return parser.parse_args()


def main():
    """Fonction principale."""
    # Analyser les arguments
    args = parse_args()
    
    # Configuration de l'environnement
    env = setup_environment()
    logger.info(f"Environnement détecté: {env.__class__.__name__}")
    
    # Charger la configuration
    if args.config:
        config = load_config_from_file(args.config)
        logger.info(f"Configuration chargée depuis {args.config}")
    else:
        config = load_default_config()
        logger.info("Configuration par défaut chargée")
    
    # Configurer l'entraînement
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    
    # Mode rapide si demandé
    if args.quick_mode:
        logger.info("Mode rapide activé, utilisation d'un sous-ensemble des données")
        config.data.max_train_tiles = 20
        config.data.max_val_tiles = 5
        config.data.max_test_tiles = 5
    
    # Créer les DataLoaders
    logger.info("Création des DataLoaders...")
    data_loaders = create_data_loaders(config)
    
    # Définir les modèles à comparer
    model_configs = [
        # U-Net de base
        {
            "name": "unet",
            "display_name": "U-Net Base",
            "params": {
                "in_channels": 1,
                "out_channels": 1,
                "init_features": 32,
                "dropout_rate": 0.2
            }
        },
        
        # U-Net avec Film
        {
            "name": "unet_film",
            "display_name": "U-Net FiLM",
            "params": {
                "in_channels": 1,
                "out_channels": 1,
                "init_features": 32,
                "dropout_rate": 0.2
            }
        },
        
        # DeepLabV3+ de base
        {
            "name": "deeplabv3_plus",
            "display_name": "DeepLabV3+ Base",
            "params": {
                "in_channels": 1,
                "out_channels": 1,
                "encoder_channels": [64, 128, 256, 512],
                "aspp_channels": 256,
                "decoder_channels": 256,
                "dropout_rate": 0.2,
                "use_cbam": False
            }
        },
        
        # DeepLabV3+ avec conditionnement par seuil
        {
            "name": "deeplabv3_plus_threshold",
            "display_name": "DeepLabV3+ Threshold",
            "params": {
                "in_channels": 1,
                "out_channels": 1,
                "encoder_channels": [64, 128, 256, 512],
                "aspp_channels": 256,
                "decoder_channels": 256,
                "threshold_encoding_dim": 128,
                "dropout_rate": 0.2,
                "use_cbam": True,
                "use_pos_encoding": True
            }
        }
    ]
    
    # Créer la comparaison de modèles
    logger.info("Initialisation de la comparaison de modèles...")
    benchmark = ModelComparison(
        model_configs=model_configs,
        base_config=config,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        test_loader=data_loaders['test'],
        output_dir=args.output_dir,
        threshold_values=[2.0, 5.0, 10.0, 15.0]
    )
    
    # Exécuter la comparaison
    logger.info("Démarrage de la comparaison des modèles...")
    results = benchmark.run()
    
    # Afficher le meilleur modèle
    best_model = benchmark.get_best_model(metric='iou')
    logger.info(f"Meilleur modèle selon IoU: {best_model.get('display_name', best_model.get('name'))}")
    
    # Sauvegarder le meilleur modèle
    output_path = Path(args.output_dir) / "best_model.pt"
    benchmark.save_best_model(output_path)
    logger.info(f"Meilleur modèle sauvegardé dans {output_path}")
    
    # Générer une visualisation des résultats
    logger.info("Génération des visualisations...")
    benchmark.visualize_results()
    
    logger.info("Benchmarking terminé!")


if __name__ == "__main__":
    main() 