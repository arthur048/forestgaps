#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de benchmark pour ForestGaps.

Ce script permet de comparer systématiquement les performances de différentes
architectures de modèles sur les mêmes données. Il génère des rapports détaillés
et des visualisations comparatives.

Auteur: Arthur VDL
"""

import os
import sys
import time
import argparse
import logging
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tabulate import tabulate

# Assurer que le package est dans le PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forestgaps.config import ConfigurationManager, load_default_config
from forestgaps.environment import setup_environment
from forestgaps.utils.errors import ForestGapsError
from forestgaps.utils.logging import setup_logging
from forestgaps.models import create_model, ModelRegistry
from forestgaps.data.loaders import create_data_loaders
from forestgaps.training import Trainer
from forestgaps.training.callbacks import (
    TensorBoardCallback,
    ModelCheckpointCallback,
    EarlyStoppingCallback
)
from forestgaps.utils.visualization import (
    visualize_metrics_comparison,
    visualize_training_curves,
    create_comparison_table
)

# ===================================================================================================
# CONFIGURATION ET INITIALISATION
# ===================================================================================================

def parse_arguments():
    """
    Analyser les arguments en ligne de commande.
    
    Returns:
        argparse.Namespace: Les arguments analysés.
    """
    parser = argparse.ArgumentParser(
        description='Script de benchmark pour ForestGaps'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Chemin vers le fichier de configuration YAML (optionnel)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['unet', 'unet_film', 'unet_cbam'],
        help='Liste des modèles à comparer'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Nombre d\'époques pour l\'entraînement'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Taille des lots pour l\'entraînement'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Répertoire de sortie pour les résultats (optionnel)'
    )
    parser.add_argument(
        '--log_level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Niveau de journalisation (défaut: INFO)'
    )
    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Exécuter en mode test rapide (échantillon réduit)'
    )
    
    return parser.parse_args()

# ===================================================================================================
# FONCTIONS DE BENCHMARK
# ===================================================================================================

def setup_benchmark_environment(config, args, logger):
    """
    Configurer l'environnement pour le benchmark.
    
    Args:
        config (ConfigurationManager): Gestionnaire de configuration.
        args (argparse.Namespace): Arguments de ligne de commande.
        logger (logging.Logger): Logger pour les messages.
        
    Returns:
        tuple: (env, dirs) - L'environnement configuré et les répertoires.
    """
    # Configurer l'environnement
    logger.info("Configuration de l'environnement...")
    env = setup_environment()
    logger.info(f"Environnement détecté: {env.name}, GPU: {env.has_gpu}")
    
    # Configurer les répertoires
    output_dir = args.output_dir or config.get("output.base_dir", os.path.expanduser("~/forestgaps_output"))
    benchmark_dir = os.path.join(output_dir, "benchmarks", f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    dirs = {
        "benchmark": benchmark_dir,
        "models": os.path.join(benchmark_dir, "models"),
        "results": os.path.join(benchmark_dir, "results"),
        "logs": os.path.join(benchmark_dir, "logs"),
        "tensorboard": os.path.join(benchmark_dir, "tensorboard"),
        "figures": os.path.join(benchmark_dir, "figures")
    }
    
    # Créer les répertoires
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Mettre à jour la configuration
    config.set("output.base_dir", benchmark_dir)
    config.set("training.epochs", args.epochs)
    config.set("training.batch_size", args.batch_size)
    
    # Mode test rapide
    if args.quick_test:
        logger.info("Mode test rapide activé")
        config.set("data.sample_size", 0.1)  # Utiliser 10% des données
        config.set("training.epochs", min(5, args.epochs))
        config.set("training.batch_size", min(8, args.batch_size))
    
    # Sauvegarder la configuration
    config_path = os.path.join(benchmark_dir, "benchmark_config.yaml")
    config.save_config(config_path)
    logger.info(f"Configuration sauvegardée dans: {config_path}")
    
    return env, dirs

def train_model_for_benchmark(model_type, config, env, data_loaders, dirs, logger):
    """
    Entraîner un modèle pour le benchmark.
    
    Args:
        model_type (str): Type de modèle à entraîner.
        config (ConfigurationManager): Gestionnaire de configuration.
        env: L'environnement d'exécution.
        data_loaders (dict): DataLoaders pour l'entraînement et la validation.
        dirs (dict): Répertoires pour les sorties.
        logger (logging.Logger): Logger pour les messages.
        
    Returns:
        dict: Résultats de l'entraînement.
    """
    logger.info(f"Entraînement du modèle: {model_type}")
    
    # Créer le modèle
    model = create_model(model_type, **config.get("model.params", {}))
    
    # Configurer les callbacks
    model_dir = os.path.join(dirs["models"], model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    tensorboard_dir = os.path.join(dirs["tensorboard"], model_type)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    callbacks = [
        TensorBoardCallback(log_dir=tensorboard_dir),
        ModelCheckpointCallback(
            save_dir=model_dir,
            save_best_only=True,
            metric_name="val_iou"
        ),
        EarlyStoppingCallback(
            patience=config.get("training.early_stopping_patience", 10),
            metric_name="val_iou",
            mode="max"
        )
    ]
    
    # Créer et configurer le Trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        test_loader=data_loaders.get('test'),
        callbacks=callbacks,
        device=env.get_device()
    )
    
    # Entraîner le modèle
    start_time = time.time()
    results = trainer.train(
        epochs=config.get("training.epochs", 50),
        gradient_clipping=config.get("training.gradient_clipping", 1.0)
    )
    training_time = time.time() - start_time
    
    # Ajouter le temps d'entraînement aux résultats
    results["training_time"] = training_time
    
    # Sauvegarder les résultats
    results_path = os.path.join(dirs["results"], f"{model_type}_results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f)
    
    logger.info(f"Entraînement terminé pour {model_type} en {training_time:.2f} secondes")
    
    return {
        "model_type": model_type,
        "results": results,
        "best_model_path": trainer.get_best_model_path(),
        "training_time": training_time
    }

def evaluate_model_for_benchmark(model_info, config, env, data_loaders, dirs, logger):
    """
    Évaluer un modèle pour le benchmark.
    
    Args:
        model_info (dict): Informations sur le modèle entraîné.
        config (ConfigurationManager): Gestionnaire de configuration.
        env: L'environnement d'exécution.
        data_loaders (dict): DataLoaders pour l'évaluation.
        dirs (dict): Répertoires pour les sorties.
        logger (logging.Logger): Logger pour les messages.
        
    Returns:
        dict: Résultats de l'évaluation.
    """
    logger.info(f"Évaluation du modèle: {model_info['model_type']}")
    
    from forestgaps.models import load_model
    from forestgaps.training.metrics import SegmentationMetrics
    
    # Charger le modèle
    model = load_model(model_info["best_model_path"])
    model.to(env.get_device())
    model.eval()
    
    # Évaluer le modèle
    test_loader = data_loaders['test']
    metrics = SegmentationMetrics(device=env.get_device())
    thresholds = config.get("data.gap_thresholds", [2.0, 5.0, 10.0, 15.0])
    
    # Mesurer le temps d'inférence
    start_time = time.time()
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, (dsm, targets) in enumerate(test_loader):
            dsm = dsm.to(env.get_device())
            num_samples += dsm.size(0)
            
            for threshold_idx, threshold in enumerate(thresholds):
                threshold_tensor = torch.tensor([threshold], device=env.get_device())
                outputs = model(dsm, threshold_tensor)
                
                target = targets[threshold_idx].to(env.get_device())
                metrics.update_by_threshold(outputs, target, threshold)
    
    inference_time = time.time() - start_time
    inference_time_per_sample = inference_time / num_samples if num_samples > 0 else 0
    
    # Calculer les métriques
    metrics_results = metrics.compute()
    
    # Ajouter les informations de temps
    metrics_results["inference_time"] = inference_time
    metrics_results["inference_time_per_sample"] = inference_time_per_sample
    
    # Sauvegarder les résultats
    results_path = os.path.join(dirs["results"], f"{model_info['model_type']}_evaluation.yaml")
    with open(results_path, "w") as f:
        yaml.dump(metrics_results, f)
    
    logger.info(f"Évaluation terminée pour {model_info['model_type']}")
    logger.info(f"Temps d'inférence total: {inference_time:.2f} secondes")
    logger.info(f"Temps d'inférence par échantillon: {inference_time_per_sample*1000:.2f} ms")
    
    return {
        "model_type": model_info["model_type"],
        "metrics": metrics_results,
        "inference_time": inference_time,
        "inference_time_per_sample": inference_time_per_sample
    }

def generate_benchmark_report(benchmark_results, dirs, logger):
    """
    Générer un rapport de benchmark complet.
    
    Args:
        benchmark_results (list): Résultats du benchmark pour chaque modèle.
        dirs (dict): Répertoires pour les sorties.
        logger (logging.Logger): Logger pour les messages.
    """
    logger.info("Génération du rapport de benchmark...")
    
    # Créer un DataFrame pour les résultats
    results_data = []
    for result in benchmark_results:
        model_type = result["model_type"]
        metrics = result["evaluation"]["metrics"]
        
        # Extraire les métriques principales
        row = {
            "Modèle": model_type,
            "IoU moyen": metrics.get("mean_iou", 0),
            "F1 moyen": metrics.get("mean_f1", 0),
            "Précision": metrics.get("mean_precision", 0),
            "Rappel": metrics.get("mean_recall", 0),
            "Temps d'entraînement (s)": result["training"]["training_time"],
            "Temps d'inférence (ms/échantillon)": result["evaluation"]["inference_time_per_sample"] * 1000
        }
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    
    # Sauvegarder les résultats en CSV
    csv_path = os.path.join(dirs["results"], "benchmark_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Créer un tableau formaté
    table = tabulate(results_df, headers="keys", tablefmt="grid", floatfmt=".4f")
    
    # Sauvegarder le tableau dans un fichier texte
    with open(os.path.join(dirs["results"], "benchmark_table.txt"), "w") as f:
        f.write(table)
    
    # Générer des visualisations
    logger.info("Génération des visualisations...")
    
    # Visualisation des métriques
    visualize_metrics_comparison(
        benchmark_results,
        metrics=["mean_iou", "mean_f1", "mean_precision", "mean_recall"],
        title="Comparaison des métriques par modèle",
        save_path=os.path.join(dirs["figures"], "metrics_comparison.png")
    )
    
    # Visualisation des courbes d'entraînement
    visualize_training_curves(
        benchmark_results,
        metrics=["train_loss", "val_loss", "val_iou"],
        title="Courbes d'entraînement par modèle",
        save_path=os.path.join(dirs["figures"], "training_curves.png")
    )
    
    # Créer un tableau de comparaison détaillé
    comparison_table = create_comparison_table(benchmark_results)
    with open(os.path.join(dirs["results"], "detailed_comparison.txt"), "w") as f:
        f.write(comparison_table)
    
    # Générer un rapport HTML
    generate_html_report(benchmark_results, dirs)
    
    logger.info(f"Rapport de benchmark généré dans: {dirs['results']}")
    logger.info(f"Visualisations disponibles dans: {dirs['figures']}")

def generate_html_report(benchmark_results, dirs):
    """
    Générer un rapport HTML détaillé.
    
    Args:
        benchmark_results (list): Résultats du benchmark pour chaque modèle.
        dirs (dict): Répertoires pour les sorties.
    """
    import base64
    from io import BytesIO
    
    # Créer le contenu HTML
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ForestGaps - Rapport de Benchmark</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }
            h1, h2, h3 { color: #2c3e50; }
            .container { max-width: 1200px; margin: 0 auto; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .chart { margin: 20px 0; max-width: 100%; }
            .model-section { margin-bottom: 40px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
            .highlight { background-color: #e8f4f8; font-weight: bold; }
            footer { margin-top: 50px; text-align: center; font-size: 0.8em; color: #777; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>ForestGaps - Rapport de Benchmark</h1>
            <p>Date: {date}</p>
            
            <h2>Résumé des performances</h2>
            {summary_table}
            
            <h2>Visualisations</h2>
            <div class="chart">
                <h3>Comparaison des métriques</h3>
                <img src="figures/metrics_comparison.png" alt="Comparaison des métriques" style="max-width:100%;">
            </div>
            
            <div class="chart">
                <h3>Courbes d'entraînement</h3>
                <img src="figures/training_curves.png" alt="Courbes d'entraînement" style="max-width:100%;">
            </div>
            
            <h2>Détails par modèle</h2>
            {model_details}
            
            <footer>
                <p>Généré par ForestGaps Benchmark Tool</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    # Créer le tableau récapitulatif
    results_data = []
    for result in benchmark_results:
        model_type = result["model_type"]
        metrics = result["evaluation"]["metrics"]
        
        row = {
            "Modèle": model_type,
            "IoU moyen": f"{metrics.get('mean_iou', 0):.4f}",
            "F1 moyen": f"{metrics.get('mean_f1', 0):.4f}",
            "Précision": f"{metrics.get('mean_precision', 0):.4f}",
            "Rappel": f"{metrics.get('mean_recall', 0):.4f}",
            "Temps d'entraînement (s)": f"{result['training']['training_time']:.2f}",
            "Temps d'inférence (ms/échantillon)": f"{result['evaluation']['inference_time_per_sample'] * 1000:.2f}"
        }
        results_data.append(row)
    
    # Créer le HTML du tableau récapitulatif
    summary_table = "<table><tr>"
    headers = list(results_data[0].keys())
    for header in headers:
        summary_table += f"<th>{header}</th>"
    summary_table += "</tr>"
    
    for row in results_data:
        summary_table += "<tr>"
        for header in headers:
            summary_table += f"<td>{row[header]}</td>"
        summary_table += "</tr>"
    summary_table += "</table>"
    
    # Créer les détails par modèle
    model_details = ""
    for result in benchmark_results:
        model_type = result["model_type"]
        metrics = result["evaluation"]["metrics"]
        training_results = result["training"]["results"]
        
        model_details += f"""
        <div class="model-section">
            <h3>{model_type}</h3>
            
            <h4>Métriques d'évaluation</h4>
            <table>
                <tr><th>Métrique</th><th>Valeur</th></tr>
                <tr><td>IoU moyen</td><td>{metrics.get('mean_iou', 0):.4f}</td></tr>
                <tr><td>F1 moyen</td><td>{metrics.get('mean_f1', 0):.4f}</td></tr>
                <tr><td>Précision</td><td>{metrics.get('mean_precision', 0):.4f}</td></tr>
                <tr><td>Rappel</td><td>{metrics.get('mean_recall', 0):.4f}</td></tr>
                <tr><td>Temps d'entraînement</td><td>{result['training']['training_time']:.2f} s</td></tr>
                <tr><td>Temps d'inférence</td><td>{result['evaluation']['inference_time_per_sample'] * 1000:.2f} ms/échantillon</td></tr>
            </table>
            
            <h4>Métriques par seuil de hauteur</h4>
            <table>
                <tr><th>Seuil</th><th>IoU</th><th>F1</th><th>Précision</th><th>Rappel</th></tr>
        """
        
        # Ajouter les métriques par seuil
        for threshold, threshold_metrics in metrics.get("by_threshold", {}).items():
            model_details += f"""
                <tr>
                    <td>{threshold}</td>
                    <td>{threshold_metrics.get('iou', 0):.4f}</td>
                    <td>{threshold_metrics.get('f1', 0):.4f}</td>
                    <td>{threshold_metrics.get('precision', 0):.4f}</td>
                    <td>{threshold_metrics.get('recall', 0):.4f}</td>
                </tr>
            """
        
        model_details += """
            </table>
        </div>
        """
    
    # Assembler le rapport HTML
    html_content = html_content.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        summary_table=summary_table,
        model_details=model_details
    )
    
    # Sauvegarder le rapport HTML
    with open(os.path.join(dirs["results"], "benchmark_report.html"), "w") as f:
        f.write(html_content)

# ===================================================================================================
# FONCTION PRINCIPALE
# ===================================================================================================

def run_benchmark(args):
    """
    Exécuter le benchmark complet.
    
    Args:
        args (argparse.Namespace): Arguments de ligne de commande.
        
    Returns:
        dict: Résultats du benchmark.
    """
    # Configurer la journalisation
    logger = setup_logging(log_level=args.log_level)
    logger.info("Démarrage du benchmark ForestGaps")
    
    try:
        # Charger la configuration
        config = load_default_config() if args.config is None else ConfigurationManager()
        
        if args.config:
            config.load_config(args.config)
            logger.info(f"Configuration chargée depuis: {args.config}")
        
        # Configurer l'environnement
        env, dirs = setup_benchmark_environment(config, args, logger)
        
        # Mettre à jour le chemin du fichier de log
        log_file = os.path.join(dirs["logs"], "benchmark.log")
        logger = setup_logging(log_level=args.log_level, log_file=log_file)
        
        # Vérifier les modèles disponibles
        available_models = ModelRegistry.list_available_models()
        logger.info(f"Modèles disponibles: {', '.join(available_models)}")
        
        # Filtrer les modèles demandés
        models_to_benchmark = [m for m in args.models if m in available_models]
        
        if not models_to_benchmark:
            logger.error(f"Aucun des modèles demandés n'est disponible: {', '.join(args.models)}")
            return {"error": "Aucun modèle valide à comparer"}
        
        logger.info(f"Modèles à comparer: {', '.join(models_to_benchmark)}")
        
        # Créer les DataLoaders
        logger.info("Création des DataLoaders...")
        data_loaders = create_data_loaders(
            config,
            batch_size=config.get("training.batch_size", args.batch_size),
            num_workers=config.get("training.num_workers", 4 if env.name == "local" else 2)
        )
        
        # Exécuter le benchmark pour chaque modèle
        benchmark_results = []
        
        for model_type in models_to_benchmark:
            logger.info(f"Benchmark du modèle: {model_type}")
            
            # Entraîner le modèle
            training_result = train_model_for_benchmark(
                model_type, config, env, data_loaders, dirs, logger
            )
            
            # Évaluer le modèle
            evaluation_result = evaluate_model_for_benchmark(
                training_result, config, env, data_loaders, dirs, logger
            )
            
            # Ajouter les résultats
            benchmark_results.append({
                "model_type": model_type,
                "training": training_result,
                "evaluation": evaluation_result
            })
        
        # Générer le rapport
        generate_benchmark_report(benchmark_results, dirs, logger)
        
        logger.info("Benchmark terminé avec succès")
        
        return {
            "results": benchmark_results,
            "output_dir": dirs["benchmark"]
        }
    
    except ForestGapsError as e:
        logger.error(f"Erreur ForestGaps: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"Erreur inattendue: {str(e)}")
        raise

# ===================================================================================================
# POINT D'ENTRÉE
# ===================================================================================================

if __name__ == "__main__":
    args = parse_arguments()
    try:
        results = run_benchmark(args)
        print(f"\nBenchmark terminé avec succès. Résultats disponibles dans: {results['output_dir']}")
    except Exception as e:
        print(f"\nErreur lors de l'exécution du benchmark: {str(e)}")
        sys.exit(1) 