#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de benchmarking complet pour les modèles ForestGaps.

Ce script exécute un benchmark complet avec tous les modèles
et configurations sur l'ensemble des données.

Usage:
    python scripts/benchmark_full.py --experiment-name "comparison_all_models"
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
logger = logging.getLogger("benchmark_full")


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Benchmark complet des modèles ForestGaps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Nom de l'expérience (sera préfixé par timestamp)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Nombre d'époques pour l'entraînement"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Taille du batch"
    )

    parser.add_argument(
        "--models",
        type=str,
        default="unet,unet_film,deeplabv3_plus,deeplabv3_plus_threshold",
        help="Modèles à comparer (séparés par des virgules)"
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        default="2.0,5.0,10.0,15.0",
        help="Seuils de hauteur à évaluer (séparés par des virgules)"
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

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Chemin vers un fichier de configuration personnalisé (optionnel)"
    )

    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Désactiver les logs TensorBoard"
    )

    parser.add_argument(
        "--save-all-checkpoints",
        action="store_true",
        help="Sauvegarder tous les checkpoints (pas seulement le meilleur)"
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
        Tuple (output_dir, log_dir, experiment_id)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{timestamp}_{experiment_name}"

    output_dir = Path(base_output_dir) / experiment_id
    log_dir = Path(base_log_dir) / experiment_id

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Créer les sous-répertoires
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)

    logger.info(f"Expérience ID: {experiment_id}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Log dir: {log_dir}")

    return output_dir, log_dir, experiment_id


def get_all_model_configs() -> list:
    """
    Retourne toutes les configurations de modèles disponibles.

    Returns:
        Liste des configurations de modèles
    """
    return [
        # U-Net de base
        {
            "name": "unet",
            "display_name": "UNet_Base",
            "params": {
                "in_channels": 1,
                "out_channels": 1,
                "init_features": 32,
                "dropout_rate": 0.2
            }
        },

        # U-Net avec FiLM (Feature-wise Linear Modulation)
        {
            "name": "unet_film",
            "display_name": "UNet_FiLM",
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
        },

        # DeepLabV3+ avec conditionnement par seuil
        {
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
        }
    ]


def filter_model_configs(all_configs: list, selected_names: list) -> list:
    """
    Filtre les configurations de modèles selon les noms sélectionnés.

    Args:
        all_configs: Toutes les configurations disponibles
        selected_names: Noms des modèles sélectionnés

    Returns:
        Liste filtrée des configurations
    """
    filtered = []
    for name in selected_names:
        matching = [c for c in all_configs if c['name'] == name]
        if matching:
            filtered.extend(matching)
        else:
            logger.warning(f"Modèle inconnu: {name}, ignoré")

    return filtered


def print_banner(text: str, char: str = "="):
    """Affiche un bandeau formaté."""
    logger.info(char * 80)
    logger.info(text)
    logger.info(char * 80)


def main():
    """Fonction principale."""
    args = parse_args()

    print_banner("BENCHMARKING COMPLET - ForestGaps")

    # Configuration de l'environnement
    logger.info("\n[1/7] Configuration de l'environnement...")
    env = setup_environment()
    logger.info(f"Environnement: {env.__class__.__name__}")
    logger.info(f"Device: {env.get_device()}")

    # Créer les répertoires d'expérience
    logger.info("\n[2/7] Création des répertoires d'expérience...")
    output_dir, log_dir, experiment_id = create_experiment_dirs(
        args.output_base_dir,
        args.log_base_dir,
        args.experiment_name
    )

    # Charger la configuration
    logger.info("\n[3/7] Chargement de la configuration...")
    if args.config:
        from forestgaps.config import load_config_from_file
        config = load_config_from_file(args.config)
        logger.info(f"Configuration chargée depuis: {args.config}")
    else:
        config = load_default_config()
        logger.info("Configuration par défaut chargée")

    # Ajuster les paramètres d'entraînement
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size

    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Optimizer: {config.training.optimizer}")

    # Sauvegarder la configuration
    config_path = output_dir / "config.yaml"
    logger.info(f"Configuration sauvegardée: {config_path}")

    # Créer les DataLoaders
    logger.info("\n[4/7] Création des DataLoaders...")
    try:
        data_loaders = create_data_loaders(config)
        logger.info(f"Train loader: {len(data_loaders['train'])} batches")
        logger.info(f"Val loader: {len(data_loaders['val'])} batches")
        logger.info(f"Test loader: {len(data_loaders['test'])} batches")
    except Exception as e:
        logger.error(f"Erreur lors de la création des DataLoaders: {e}")
        logger.error("Vérifiez que les données sont présentes dans 'data/'")
        return 1

    # Préparer les configurations des modèles
    logger.info("\n[5/7] Préparation des modèles...")
    all_configs = get_all_model_configs()
    model_names = [m.strip() for m in args.models.split(',')]
    model_configs = filter_model_configs(all_configs, model_names)

    if not model_configs:
        logger.error("Aucune configuration de modèle valide")
        return 1

    logger.info(f"Modèles à comparer ({len(model_configs)}):")
    for mc in model_configs:
        logger.info(f"  - {mc['display_name']} ({mc['name']})")

    # Parser les seuils
    thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    logger.info(f"Seuils de hauteur: {thresholds} mètres")

    # Informations sur le temps estimé
    logger.info("\n[6/7] Estimation du temps de calcul...")
    n_models = len(model_configs)
    n_epochs = args.epochs
    estimated_time_per_epoch = 2  # minutes (approximatif)
    total_estimated_minutes = n_models * n_epochs * estimated_time_per_epoch
    logger.info(f"Temps estimé: ~{total_estimated_minutes} minutes (~{total_estimated_minutes/60:.1f}h)")
    logger.info("Note: Ceci est une estimation approximative")

    # Confirmer avant de lancer
    logger.info("\n" + "-"*80)
    logger.info("Configuration du benchmark:")
    logger.info(f"  - Expérience: {experiment_id}")
    logger.info(f"  - Modèles: {n_models}")
    logger.info(f"  - Epochs: {n_epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Seuils: {len(thresholds)}")
    logger.info(f"  - TensorBoard: {'Désactivé' if args.no_tensorboard else 'Activé (http://localhost:6006)'}")
    logger.info("-"*80)

    # Créer et exécuter le benchmark
    logger.info("\n[7/7] Exécution du benchmark...")
    print_banner("Début de l'entraînement", char="-")

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

        print_banner("BENCHMARK TERMINÉ AVEC SUCCÈS !")

        # Afficher les résultats principaux
        logger.info("\nRésultats principaux:")
        logger.info("-"*80)

        # Meilleurs modèles par métrique
        logger.info("\nMeilleurs modèles:")
        for metric in ['iou', 'f1', 'precision', 'recall']:
            best_model = benchmark.get_best_model(metric=metric)
            if best_model:
                name = best_model.get('display_name', best_model.get('name'))
                logger.info(f"  - {metric.upper():12s}: {name}")

        # Modèle le plus rapide
        best_models = results.get('best_models', {})
        if 'training_time' in best_models:
            logger.info(f"  - {'TIME':12s}: {best_models['training_time']}")

        # Sauvegarder le meilleur modèle
        logger.info("\nSauvegarde du meilleur modèle...")
        best_model_path = output_dir / "best_model.pt"
        saved_path = benchmark.save_best_model(best_model_path)
        if saved_path:
            logger.info(f"Meilleur modèle: {saved_path}")

        # Résumé des chemins
        logger.info("\n" + "="*80)
        logger.info("RÉSULTATS DISPONIBLES DANS:")
        logger.info("="*80)
        logger.info(f"📁 Outputs:      {output_dir}")
        logger.info(f"📊 Logs:         {log_dir}")
        logger.info(f"📈 TensorBoard:  http://localhost:6006")
        logger.info(f"📄 Rapport HTML: {output_dir}/reports/benchmark_report.html")
        logger.info(f"📋 Résultats:    {output_dir}/benchmark_results.json")
        logger.info(f"🏆 Meilleur:     {best_model_path}")
        logger.info("="*80)

        # Conseils pour la suite
        logger.info("\nPROCHAINES ÉTAPES:")
        logger.info("1. Visualiser dans TensorBoard: http://localhost:6006")
        logger.info("2. Consulter le rapport HTML pour l'analyse détaillée")
        logger.info("3. Évaluer le meilleur modèle sur données externes:")
        logger.info(f"   python scripts/evaluate_external.py --model {best_model_path}")
        logger.info("")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nBenchmark interrompu par l'utilisateur")
        logger.info("Les résultats partiels sont disponibles dans:")
        logger.info(f"  - {output_dir}")
        return 130

    except Exception as e:
        logger.error(f"\nErreur lors du benchmark: {e}", exc_info=True)
        logger.error("\nVérifiez:")
        logger.error("  1. Les données sont présentes dans data/")
        logger.error("  2. Le GPU est disponible (nvidia-smi)")
        logger.error("  3. La mémoire est suffisante")
        logger.error("  4. Les logs pour plus de détails")
        return 1


if __name__ == "__main__":
    sys.exit(main())
