#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test rapide pour le benchmarking des modèles ForestGaps.

Ce script permet de tester rapidement le workflow de benchmarking
avec un sous-ensemble limité de données et peu d'époques.

Usage:
    python scripts/benchmark_quick_test.py --experiment-name "test_run"
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forestgaps.config import load_default_config
from forestgaps.environment import setup_environment
from forestgaps.data.loaders import create_data_loaders
from forestgaps.benchmarking import ModelComparison

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("benchmark_quick_test")


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Test rapide du benchmarking ForestGaps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="quick_test",
        help="Nom de l'expérience (sera préfixé par timestamp)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Nombre d'époques pour l'entraînement rapide"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Taille du batch"
    )

    parser.add_argument(
        "--max-train-tiles",
        type=int,
        default=20,
        help="Nombre maximum de tuiles d'entraînement"
    )

    parser.add_argument(
        "--max-val-tiles",
        type=int,
        default=5,
        help="Nombre maximum de tuiles de validation"
    )

    parser.add_argument(
        "--models",
        type=str,
        default="unet,unet_film",
        help="Modèles à tester (séparés par des virgules)"
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        default="5.0,10.0",
        help="Seuils de hauteur à tester (séparés par des virgules)"
    )

    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="outputs/benchmarks",
        help="Répertoire de base pour les outputs"
    )

    parser.add_argument(
        "--log-base-dir",
        type=str,
        default="logs/benchmarks",
        help="Répertoire de base pour les logs"
    )

    return parser.parse_args()


def create_experiment_dirs(base_output_dir: str, base_log_dir: str, experiment_name: str):
    """
    Crée les répertoires pour l'expérience avec timestamp.

    Args:
        base_output_dir: Répertoire de base pour outputs
        base_log_dir: Répertoire de base pour logs
        experiment_name: Nom de l'expérience

    Returns:
        Tuple (output_dir, log_dir)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{timestamp}_{experiment_name}"

    output_dir = Path(base_output_dir) / experiment_id
    log_dir = Path(base_log_dir) / experiment_id

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Expérience ID: {experiment_id}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Log dir: {log_dir}")

    return output_dir, log_dir, experiment_id


def get_model_configs(model_names: list) -> list:
    """
    Génère les configurations des modèles à partir des noms.

    Args:
        model_names: Liste des noms de modèles

    Returns:
        Liste des configurations de modèles
    """
    model_configs = []

    for name in model_names:
        if name == "unet":
            model_configs.append({
                "name": "unet",
                "display_name": "UNet_Base",
                "params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "init_features": 32,
                    "dropout_rate": 0.2
                }
            })

        elif name == "unet_film":
            model_configs.append({
                "name": "unet_film",
                "display_name": "UNet_FiLM",
                "params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "init_features": 32,
                    "dropout_rate": 0.2
                }
            })

        elif name == "deeplabv3_plus":
            model_configs.append({
                "name": "deeplabv3_plus",
                "display_name": "DeepLabV3+_Base",
                "params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "encoder_channels": [64, 128, 256, 512],
                    "aspp_channels": 256,
                    "decoder_channels": 256,
                    "dropout_rate": 0.2,
                    "use_cbam": False
                }
            })

        elif name == "deeplabv3_plus_threshold":
            model_configs.append({
                "name": "deeplabv3_plus_threshold",
                "display_name": "DeepLabV3+_Threshold",
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
            })

        else:
            logger.warning(f"Modèle inconnu: {name}, ignoré")

    return model_configs


def main():
    """Fonction principale."""
    args = parse_args()

    logger.info("="*80)
    logger.info("BENCHMARKING QUICK TEST - ForestGaps")
    logger.info("="*80)

    # Configuration de l'environnement
    logger.info("\n[1/6] Configuration de l'environnement...")
    env = setup_environment()
    logger.info(f"Environnement détecté: {env.__class__.__name__}")
    logger.info(f"Device: {env.get_device()}")

    # Créer les répertoires d'expérience
    logger.info("\n[2/6] Création des répertoires d'expérience...")
    output_dir, log_dir, experiment_id = create_experiment_dirs(
        args.output_base_dir,
        args.log_base_dir,
        args.experiment_name
    )

    # Charger et configurer la configuration
    logger.info("\n[3/6] Chargement de la configuration...")
    config = load_default_config()

    # Ajuster pour test rapide
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.data.max_train_tiles = args.max_train_tiles
    config.data.max_val_tiles = args.max_val_tiles
    config.data.max_test_tiles = args.max_val_tiles  # Même taille que val

    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Max train tiles: {config.data.max_train_tiles}")
    logger.info(f"Max val tiles: {config.data.max_val_tiles}")

    # Sauvegarder la configuration
    config_path = output_dir / "config.yaml"
    # Note: Ajouter méthode save si nécessaire dans config
    logger.info(f"Configuration sauvegardée dans: {config_path}")

    # Créer les DataLoaders
    logger.info("\n[4/6] Création des DataLoaders...")
    try:
        data_loaders = create_data_loaders(config)
        logger.info(f"Train loader: {len(data_loaders['train'])} batches")
        logger.info(f"Val loader: {len(data_loaders['val'])} batches")
        logger.info(f"Test loader: {len(data_loaders['test'])} batches")
    except Exception as e:
        logger.error(f"Erreur lors de la création des DataLoaders: {e}")
        logger.error("Vérifiez que les données sont présentes dans le répertoire 'data/'")
        return 1

    # Préparer les configurations des modèles
    logger.info("\n[5/6] Préparation des modèles...")
    model_names = [m.strip() for m in args.models.split(',')]
    model_configs = get_model_configs(model_names)

    if not model_configs:
        logger.error("Aucune configuration de modèle valide")
        return 1

    logger.info(f"Modèles à comparer: {[m['display_name'] for m in model_configs]}")

    # Parser les seuils
    thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    logger.info(f"Seuils de hauteur: {thresholds}")

    # Créer et exécuter le benchmark
    logger.info("\n[6/6] Exécution du benchmark...")
    logger.info("-"*80)

    benchmark = ModelComparison(
        model_configs=model_configs,
        base_config=config,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        test_loader=data_loaders['test'],
        output_dir=output_dir,
        threshold_values=thresholds
    )

    try:
        results = benchmark.run()

        logger.info("\n" + "="*80)
        logger.info("BENCHMARK TERMINÉ !")
        logger.info("="*80)

        # Afficher les meilleurs modèles
        logger.info("\nMeilleurs modèles par métrique:")
        for metric in ['iou', 'f1']:
            best_model = benchmark.get_best_model(metric=metric)
            if best_model:
                logger.info(f"  - {metric.upper()}: {best_model.get('display_name', best_model.get('name'))}")

        # Sauvegarder le meilleur modèle
        best_model_path = output_dir / "best_model.pt"
        saved_path = benchmark.save_best_model(best_model_path)
        if saved_path:
            logger.info(f"\nMeilleur modèle sauvegardé: {saved_path}")

        # Afficher les chemins importants
        logger.info("\n" + "-"*80)
        logger.info("Résultats disponibles dans:")
        logger.info(f"  - Outputs: {output_dir}")
        logger.info(f"  - Logs: {log_dir}")
        logger.info(f"  - TensorBoard: http://localhost:6006 (si lancé)")
        logger.info(f"  - Rapport HTML: {output_dir}/reports/benchmark_report.html")
        logger.info("-"*80)

        return 0

    except Exception as e:
        logger.error(f"\nErreur lors du benchmark: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
