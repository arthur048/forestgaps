#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de workflow complet pour ForestGaps-DL.

Ce script implémente le workflow de bout en bout pour la détection de trouées forestières,
incluant la préparation des données, l'entraînement d'un modèle et l'évaluation des résultats.
Le script est optimisé pour fonctionner dans les environnements Colab et locaux.

Auteur: Arthur VDL
"""

import os
import sys
import time
import argparse
import logging
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from datetime import datetime

# Assurer que le package est dans le PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forestgaps_dl.config import ConfigurationManager, load_default_config
from forestgaps_dl.environment import setup_environment, is_colab_environment
from forestgaps_dl.utils.errors import ForestGapsError
from forestgaps_dl.utils.profiling import Profiler
from forestgaps_dl.utils.logging import setup_logging

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
        description='Script de workflow complet pour ForestGaps-DL'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Chemin vers le fichier de configuration YAML (optionnel)'
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'data_prep', 'train', 'eval', 'test'],
        default='full',
        help='Mode d\'exécution (défaut: full)'
    )
    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Exécuter en mode test rapide (échantillon réduit)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='unet',
        help='Type de modèle à utiliser (défaut: unet)'
    )
    parser.add_argument(
        '--log_level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Niveau de journalisation (défaut: INFO)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Graine aléatoire pour la reproductibilité (défaut: 42)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Répertoire de sortie pour les résultats (optionnel)'
    )
    parser.add_argument(
        '--profiling',
        action='store_true',
        help='Activer le profilage des performances'
    )
    
    return parser.parse_args()

def set_seed(seed):
    """
    Définir les graines aléatoires pour la reproductibilité.
    
    Args:
        seed (int): La graine aléatoire à utiliser.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_workspace(config):
    """
    Configurer les répertoires de travail selon la configuration.
    
    Args:
        config (ConfigurationManager): Gestionnaire de configuration.
        
    Returns:
        dict: Dictionnaire des chemins configurés.
    """
    # Extraire les répertoires de base de la configuration
    base_dir = config.get("data.base_dir", os.path.expanduser("~/forestgaps_data"))
    output_dir = config.get("output.base_dir", os.path.expanduser("~/forestgaps_output"))
    
    # Créer les sous-répertoires nécessaires
    dirs = {
        "base": base_dir,
        "raw": os.path.join(base_dir, "raw"),
        "processed": os.path.join(base_dir, "processed"),
        "tiles": os.path.join(base_dir, "tiles"),
        "output": output_dir,
        "models": os.path.join(output_dir, "models"),
        "logs": os.path.join(output_dir, "logs"),
        "results": os.path.join(output_dir, "results"),
        "tensorboard": os.path.join(output_dir, "tensorboard"),
        "tmp": os.path.join(output_dir, "tmp")
    }
    
    # Créer les répertoires s'ils n'existent pas
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs

# ===================================================================================================
# PRÉPARATION DES DONNÉES
# ===================================================================================================

def prepare_data(config, env, logger):
    """
    Exécuter la préparation des données.
    
    Args:
        config (ConfigurationManager): Gestionnaire de configuration.
        env: L'environnement d'exécution.
        logger (logging.Logger): Logger pour les messages.
        
    Returns:
        dict: Résultat de la préparation des données.
    """
    logger.info("Début de la préparation des données...")
    
    from forestgaps_dl.data.preprocessing import preprocess_rasters
    from forestgaps_dl.data.generation import generate_tiles
    from forestgaps_dl.data.normalization import compute_normalization_stats
    
    # Étape 1: Prétraitement des rasters
    logger.info("Prétraitement des rasters...")
    raster_pairs = preprocess_rasters(
        dsm_dir=config.get("data.dsm_dir"),
        chm_dir=config.get("data.chm_dir"),
        output_dir=config.get("data.processed_dir"),
        force_reprocessing=config.get("data.force_reprocessing", False)
    )
    
    # Étape 2: Génération des masques de trouées
    logger.info("Génération des masques de trouées...")
    thresholds = config.get("data.gap_thresholds", [2.0, 5.0, 10.0, 15.0])
    gap_masks = generate_tiles(
        raster_pairs=raster_pairs,
        thresholds=thresholds,
        tile_size=config.get("data.tile_size", 256),
        overlap=config.get("data.overlap", 0.2),
        output_dir=config.get("data.tiles_dir"),
        min_valid_ratio=config.get("data.min_valid_ratio", 0.7),
        force_regenerate=config.get("data.force_regenerate", False)
    )
    
    # Étape 3: Calculer les statistiques de normalisation
    logger.info("Calcul des statistiques de normalisation...")
    normalization_stats = compute_normalization_stats(
        tiles_dir=config.get("data.tiles_dir"),
        save_path=os.path.join(config.get("data.processed_dir"), "normalization_stats.pt")
    )
    
    logger.info("Préparation des données terminée.")
    
    return {
        "raster_pairs": raster_pairs,
        "gap_masks": gap_masks,
        "normalization_stats": normalization_stats
    }

# ===================================================================================================
# ENTRAINEMENT DU MODÈLE
# ===================================================================================================

def train_model(config, env, logger, profiler=None):
    """
    Entraîner un modèle de segmentation.
    
    Args:
        config (ConfigurationManager): Gestionnaire de configuration.
        env: L'environnement d'exécution.
        logger (logging.Logger): Logger pour les messages.
        profiler (Profiler, optional): Profiler de performances.
        
    Returns:
        dict: Résultats de l'entraînement et chemins des fichiers de modèles.
    """
    logger.info("Début de l'entraînement du modèle...")
    
    from forestgaps_dl.data.loaders import create_data_loaders
    from forestgaps_dl.models import create_model
    from forestgaps_dl.training import Trainer
    from forestgaps_dl.training.callbacks import (
        TensorBoardCallback,
        ModelCheckpointCallback,
        EarlyStoppingCallback,
        LearningRateSchedulerCallback,
        ProfilingCallback
    )
    
    # Activer le profilage si demandé
    if profiler:
        profiler.start_section("model_training")
    
    # Étape 1: Créer les DataLoaders optimisés
    logger.info("Création des DataLoaders optimisés...")
    data_loaders = create_data_loaders(
        config,
        batch_size=config.get("training.batch_size", 32),
        num_workers=config.get("training.num_workers", 4 if env.name == "local" else 2),
        persistent_workers=config.get("training.persistent_workers", True)
    )
    
    # Étape 2: Créer le modèle
    logger.info(f"Création du modèle: {config.get('model.type', 'unet')}...")
    model = create_model(
        model_type=config.get("model.type", "unet"),
        **config.get("model.params", {})
    )
    
    # Étape 3: Configurer les callbacks
    logger.info("Configuration des callbacks...")
    callbacks = [
        TensorBoardCallback(log_dir=config.get("training.tensorboard_dir")),
        ModelCheckpointCallback(
            save_dir=config.get("training.model_dir"),
            save_best_only=True,
            metric_name="val_iou"
        ),
        EarlyStoppingCallback(
            patience=config.get("training.early_stopping_patience", 10),
            metric_name="val_iou",
            mode="max"
        ),
        LearningRateSchedulerCallback(
            mode=config.get("training.lr_scheduler.mode", "cosine"),
            **config.get("training.lr_scheduler.params", {})
        )
    ]
    
    # Ajouter le callback de profilage si activé
    if profiler:
        callbacks.append(ProfilingCallback(profiler=profiler))
    
    # Étape 4: Créer et configurer le Trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        test_loader=data_loaders.get('test'),
        callbacks=callbacks,
        device=env.get_device()
    )
    
    # Étape 5: Entraîner le modèle
    logger.info("Démarrage de l'entraînement...")
    results = trainer.train(
        epochs=config.get("training.epochs", 100),
        gradient_clipping=config.get("training.gradient_clipping", 1.0)
    )
    
    # Étape 6: Sauvegarder les résultats
    logger.info("Sauvegarde des résultats d'entraînement...")
    trainer.save_training_summary(config.get("training.summary_path"))
    
    # Terminer le profilage si actif
    if profiler:
        profiler.end_section("model_training")
    
    logger.info("Entraînement du modèle terminé.")
    
    return {
        "training_results": results,
        "model_paths": trainer.get_model_paths(),
        "best_model_path": trainer.get_best_model_path()
    }

# ===================================================================================================
# ÉVALUATION ET VISUALISATION
# ===================================================================================================

def evaluate_model(config, train_results, env, logger):
    """
    Évaluer le modèle entraîné.
    
    Args:
        config (ConfigurationManager): Gestionnaire de configuration.
        train_results (dict): Résultats de l'entraînement.
        env: L'environnement d'exécution.
        logger (logging.Logger): Logger pour les messages.
        
    Returns:
        dict: Résultats de l'évaluation.
    """
    logger.info("Début de l'évaluation du modèle...")
    
    from forestgaps_dl.models import load_model
    from forestgaps_dl.data.loaders import create_data_loaders
    from forestgaps_dl.training.metrics import SegmentationMetrics
    from forestgaps_dl.utils.visualization import (
        visualize_predictions,
        visualize_metrics_by_threshold,
        create_metrics_tables,
        visualize_confusion_matrix
    )
    
    # Étape 1: Charger le meilleur modèle
    logger.info("Chargement du meilleur modèle...")
    model = load_model(train_results["best_model_path"])
    model.to(env.get_device())
    model.eval()
    
    # Étape 2: Créer le DataLoader de test
    logger.info("Chargement des données de test...")
    test_loader = create_data_loaders(
        config, 
        subset="test",
        batch_size=config.get("evaluation.batch_size", 16),
        num_workers=config.get("evaluation.num_workers", 4 if env.name == "local" else 2)
    )['test']
    
    # Étape 3: Évaluer le modèle
    logger.info("Évaluation du modèle sur les données de test...")
    metrics = SegmentationMetrics(device=env.get_device())
    thresholds = config.get("data.gap_thresholds", [2.0, 5.0, 10.0, 15.0])
    results_by_threshold = {}
    
    with torch.no_grad():
        for batch_idx, (dsm, targets) in enumerate(test_loader):
            dsm = dsm.to(env.get_device())
            
            for threshold_idx, threshold in enumerate(thresholds):
                threshold_tensor = torch.tensor([threshold], device=env.get_device())
                outputs = model(dsm, threshold_tensor)
                
                target = targets[threshold_idx].to(env.get_device())
                metrics.update_by_threshold(outputs, target, threshold)
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Évaluation: {batch_idx}/{len(test_loader)} lots traités")
    
    # Calculer les métriques finales
    metrics_results = metrics.compute()
    
    # Étape 4: Visualiser les résultats
    logger.info("Visualisation des résultats...")
    results_dir = config.get("evaluation.results_dir", os.path.join(config.get("output.base_dir"), "results"))
    os.makedirs(results_dir, exist_ok=True)
    
    # Générer les visualisations
    visualize_predictions(
        model=model,
        data_loader=test_loader,
        device=env.get_device(),
        thresholds=thresholds,
        save_dir=os.path.join(results_dir, "predictions"),
        num_samples=config.get("evaluation.num_visualizations", 5)
    )
    
    visualize_metrics_by_threshold(
        metrics=metrics_results,
        title="Métriques par seuil de hauteur",
        save_path=os.path.join(results_dir, "metrics_by_threshold.png")
    )
    
    visualize_confusion_matrix(
        confusion_data=metrics.compute_confusion_matrix(),
        title="Matrice de confusion",
        save_path=os.path.join(results_dir, "confusion_matrix.png")
    )
    
    # Créer un tableau récapitulatif des métriques
    metrics_table = create_metrics_tables(metrics_results)
    
    # Sauvegarder les métriques dans un fichier YAML
    with open(os.path.join(results_dir, "evaluation_metrics.yaml"), "w") as f:
        yaml.dump(metrics_results, f)
    
    logger.info("Évaluation du modèle terminée.")
    
    return {
        "metrics": metrics_results,
        "confusion_matrix": metrics.compute_confusion_matrix(),
        "visualizations_dir": os.path.join(results_dir, "predictions")
    }

# ===================================================================================================
# FONCTIONS D'OPTIMISATION
# ===================================================================================================

def optimize_resources(env, config, logger):
    """
    Optimiser l'utilisation des ressources en fonction de l'environnement.
    
    Args:
        env: L'environnement d'exécution.
        config (ConfigurationManager): Gestionnaire de configuration.
        logger (logging.Logger): Logger pour les messages.
    """
    logger.info(f"Optimisation des ressources pour l'environnement: {env.name}")
    
    from forestgaps_dl.utils.optimization import (
        optimize_cuda_operations,
        optimize_dataloader_params,
        benchmark_transfers
    )
    
    # Optimiser les opérations CUDA si disponible
    if env.has_gpu:
        logger.info("Optimisation des opérations CUDA...")
        optimize_cuda_operations()
    
    # Configuration spécifique à Colab
    if env.name == "colab":
        logger.info("Application des optimisations spécifiques à Colab...")
        # Limiter l'utilisation de la mémoire
        if env.has_gpu:
            import torch
            torch.cuda.empty_cache()
            
            # Configurer pour la fraction de mémoire GPU à utiliser
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Limiter le nombre de workers pour les DataLoaders
        config.set("training.num_workers", min(2, config.get("training.num_workers", 4)))
        
    # Optimiser les paramètres du DataLoader
    logger.info("Optimisation des paramètres du DataLoader...")
    optimal_params = optimize_dataloader_params(
        batch_size=config.get("training.batch_size", 32),
        max_workers=config.get("training.num_workers", 4)
    )
    
    # Mettre à jour la configuration avec les paramètres optimaux
    config.set("training.num_workers", optimal_params["num_workers"])
    config.set("training.prefetch_factor", optimal_params["prefetch_factor"])
    
    # Tester les méthodes de transfert CPU/GPU
    if env.has_gpu:
        logger.info("Test des méthodes de transfert CPU/GPU...")
        transfer_results = benchmark_transfers()
        best_method = transfer_results["best_method"]
        logger.info(f"Meilleure méthode de transfert: {best_method}")
        
        # Mettre à jour la configuration avec la meilleure méthode
        config.set("training.pin_memory", best_method["pin_memory"])
        config.set("training.non_blocking", best_method["non_blocking"])

# ===================================================================================================
# WORKFLOW PRINCIPAL
# ===================================================================================================

def run_workflow(args):
    """
    Exécuter le workflow complet ou partiel selon les arguments.
    
    Args:
        args (argparse.Namespace): Arguments de ligne de commande.
        
    Returns:
        dict: Résultats du workflow.
    """
    # Démarrer le chronomètre
    start_time = time.time()
    
    # Configurer la journalisation
    logger = setup_logging(
        log_level=args.log_level,
        log_file=None  # Sera configuré après l'initialisation de la configuration
    )
    
    # Définir les graines aléatoires pour la reproductibilité
    set_seed(args.seed)
    logger.info(f"Graine aléatoire définie: {args.seed}")
    
    try:
        # Configurer l'environnement
        logger.info("Configuration de l'environnement...")
        env = setup_environment()
        logger.info(f"Environnement détecté: {env.name}, GPU: {env.has_gpu}")
        
        # Charger la configuration
        logger.info("Chargement de la configuration...")
        config = load_default_config() if args.config is None else ConfigurationManager()
        
        if args.config:
            config.load_config(args.config)
            logger.info(f"Configuration chargée depuis: {args.config}")
        
        # Appliquer les paramètres de ligne de commande à la configuration
        if args.output_dir:
            config.set("output.base_dir", args.output_dir)
        
        if args.model_type:
            config.set("model.type", args.model_type)
        
        # Mode test rapide
        if args.quick_test:
            logger.info("Mode test rapide activé")
            config.set("data.sample_size", 0.1)  # Utiliser 10% des données
            config.set("training.epochs", 5)
            config.set("training.batch_size", 8)
        
        # Configurer les répertoires de travail
        dirs = setup_workspace(config)
        
        # Mettre à jour le chemin du fichier de log
        log_file = os.path.join(dirs["logs"], f"forestgaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logger = setup_logging(log_level=args.log_level, log_file=log_file)
        
        # Créer un profiler si demandé
        profiler = Profiler(enabled=args.profiling, output_dir=dirs["logs"]) if args.profiling else None
        
        # Optimiser l'utilisation des ressources
        optimize_resources(env, config, logger)
        
        # Sauvegarder la configuration actuelle
        config_path = os.path.join(dirs["output"], "config.yaml")
        config.save_config(config_path)
        logger.info(f"Configuration sauvegardée dans: {config_path}")
        
        # Exécuter les étapes selon le mode
        results = {}
        
        if args.mode in ['full', 'data_prep']:
            # Étape de préparation des données
            if profiler:
                profiler.start_section("data_preparation")
            
            data_results = prepare_data(config, env, logger)
            results["data"] = data_results
            
            if profiler:
                profiler.end_section("data_preparation")
        
        if args.mode in ['full', 'train']:
            # Étape d'entraînement du modèle
            train_results = train_model(config, env, logger, profiler)
            results["training"] = train_results
        
        if args.mode in ['full', 'eval']:
            # Étape d'évaluation
            if profiler:
                profiler.start_section("evaluation")
            
            # S'assurer que les résultats d'entraînement sont disponibles
            if "training" not in results:
                logger.info("Chargement des résultats d'entraînement précédents...")
                # Charger le modèle à partir du chemin configuré
                results["training"] = {
                    "best_model_path": config.get("evaluation.model_path")
                }
            
            eval_results = evaluate_model(config, results["training"], env, logger)
            results["evaluation"] = eval_results
            
            if profiler:
                profiler.end_section("evaluation")
        
        # Calculer et afficher le temps total d'exécution
        total_time = time.time() - start_time
        logger.info(f"Workflow terminé en {total_time:.2f} secondes")
        
        # Sauvegarder les rapports de profilage
        if profiler:
            profiler.save_reports(os.path.join(dirs["logs"], "profiling_report"))
        
        return results
    
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
        results = run_workflow(args)
        print(f"\nWorkflow '{args.mode}' terminé avec succès.")
    except Exception as e:
        print(f"\nErreur lors de l'exécution du workflow: {str(e)}")
        sys.exit(1) 