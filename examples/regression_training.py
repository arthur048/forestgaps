#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exemple d'entraînement d'un modèle de régression pour prédire le CHM à partir du DSM.

Ce script montre comment utiliser les modèles U-Net de régression pour prédire
des valeurs continues (CHM) à partir du DSM, plutôt que des masques binaires.
"""

import os
import argparse
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from forestgaps_dl.config import load_config_from_file, load_default_config
from forestgaps_dl.environment import setup_environment
from forestgaps_dl.models import create_model
from forestgaps_dl.data.datasets.regression_dataset import (
    create_regression_dataset, 
    create_regression_dataloader,
    split_regression_dataset
)
from forestgaps_dl.training.metrics.regression import RegressionMetrics
from forestgaps_dl.training.loss.regression import MSELoss, CombinedRegressionLoss
from forestgaps_dl.utils.visualization.plots import plot_regression_results


# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("regression_example")


def parse_args():
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraînement d'un modèle de régression ForestGaps-DL")
    
    parser.add_argument("--config", type=str, default=None,
                        help="Chemin vers le fichier de configuration.")
    parser.add_argument("--output-dir", type=str, default="regression_results",
                        help="Répertoire de sortie pour les résultats.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Nombre d'époques pour l'entraînement.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Taille du batch pour l'entraînement.")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Taux d'apprentissage pour l'optimiseur.")
    parser.add_argument("--model-type", type=str, default="regression_unet",
                        choices=["regression_unet", "regression_unet_threshold"],
                        help="Type de modèle à utiliser.")
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
    
    # Configurer le répertoire de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurer les sous-répertoires
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    results_dir = output_dir / "results"
    
    for directory in [checkpoints_dir, logs_dir, results_dir]:
        directory.mkdir(exist_ok=True)
    
    # Créer le writer TensorBoard
    writer = SummaryWriter(logs_dir)
    
    # Configurer le device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Utilisation du device: {device}")
    
    # Simuler des données pour l'exemple
    # Dans un cas réel, vous chargeriez vos données à partir de fichiers
    logger.info("Préparation des données...")
    
    # Simuler des fichiers DSM et CHM
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Créer des données synthétiques si elles n'existent pas déjà
    dsm_dir = data_dir / "dsm"
    chm_dir = data_dir / "chm"
    dsm_dir.mkdir(exist_ok=True)
    chm_dir.mkdir(exist_ok=True)
    
    # Nombre d'échantillons à générer
    n_samples = 20 if args.quick_mode else 100
    
    # Générer des données synthétiques
    dsm_files = []
    chm_files = []
    thresholds = []
    
    for i in range(n_samples):
        # Créer un DSM synthétique (terrain avec quelques variations)
        dsm_tile = np.random.normal(100, 10, (256, 256)).astype(np.float32)
        dsm_tile = np.clip(dsm_tile, 50, 150)  # Limiter les valeurs
        
        # Créer un CHM synthétique (hauteur des arbres, corrélée au DSM mais avec des variations)
        base_height = np.random.uniform(5, 15)  # Hauteur de base aléatoire
        chm_tile = np.zeros((256, 256), dtype=np.float32)
        
        # Ajouter des arbres comme des gaussiennes
        for _ in range(50):
            x = np.random.randint(0, 256)
            y = np.random.randint(0, 256)
            height = np.random.normal(base_height, 3)
            radius = np.random.uniform(5, 15)
            
            # Créer une gaussienne 2D pour représenter un arbre
            y_grid, x_grid = np.mgrid[0:256, 0:256]
            tree = height * np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * radius**2))
            
            # Ajouter l'arbre au CHM
            chm_tile += tree
        
        # S'assurer que le CHM est positif
        chm_tile = np.clip(chm_tile, 0, 50)
        
        # Sauvegarder les tuiles
        dsm_path = dsm_dir / f"dsm_tile_{i:03d}.npy"
        chm_path = chm_dir / f"chm_tile_{i:03d}.npy"
        
        np.save(dsm_path, dsm_tile)
        np.save(chm_path, chm_tile)
        
        dsm_files.append(str(dsm_path))
        chm_files.append(str(chm_path))
        thresholds.append(base_height)  # Utiliser la hauteur de base comme seuil
    
    # Diviser les données en ensembles d'entraînement, validation et test
    train_dsm, train_chm, val_dsm, val_chm, test_dsm, test_chm = split_regression_dataset(
        dsm_files=dsm_files,
        chm_files=chm_files
    )
    
    # Créer les seuils pour chaque ensemble
    train_thresholds = thresholds[:len(train_dsm)]
    val_thresholds = thresholds[len(train_dsm):len(train_dsm) + len(val_dsm)]
    test_thresholds = thresholds[len(train_dsm) + len(val_dsm):]
    
    # Créer les datasets
    logger.info("Création des datasets...")
    train_dataset = create_regression_dataset(
        dsm_files=train_dsm,
        chm_files=train_chm,
        thresholds=train_thresholds if args.model_type == "regression_unet_threshold" else None,
        transform_config={"is_train": True, "prob": 0.5},
        normalize=True
    )
    
    val_dataset = create_regression_dataset(
        dsm_files=val_dsm,
        chm_files=val_chm,
        thresholds=val_thresholds if args.model_type == "regression_unet_threshold" else None,
        transform_config={"is_train": False},
        normalize=True,
        stats=train_dataset.stats  # Utiliser les mêmes stats que l'entraînement
    )
    
    test_dataset = create_regression_dataset(
        dsm_files=test_dsm,
        chm_files=test_chm,
        thresholds=test_thresholds if args.model_type == "regression_unet_threshold" else None,
        transform_config={"is_train": False},
        normalize=True,
        stats=train_dataset.stats  # Utiliser les mêmes stats que l'entraînement
    )
    
    # Créer les dataloaders
    train_loader = create_regression_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = create_regression_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = create_regression_dataloader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Créer le modèle
    logger.info(f"Création du modèle de type {args.model_type}...")
    model = create_model(args.model_type)
    model.to(device)
    
    # Créer la fonction de perte
    criterion = CombinedRegressionLoss(mse_weight=0.7, mae_weight=0.3)
    
    # Créer l'optimiseur
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Créer le scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Créer les métriques
    metrics = RegressionMetrics(device=device)
    
    # Boucle d'entraînement
    logger.info(f"Début de l'entraînement pour {args.epochs} époques...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Mode entraînement
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Extraire les données
            if args.model_type == "regression_unet_threshold":
                inputs, thresholds, targets = [t.to(device) for t in batch]
            else:
                inputs, targets = [t.to(device) for t in batch]
                thresholds = None
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs, thresholds)
            
            # Calculer la perte
            loss = criterion(outputs, targets, thresholds)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumuler la perte
            train_loss += loss.item()
            
            # Afficher la progression
            if (batch_idx + 1) % 5 == 0:
                logger.info(f"Époque {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculer la perte moyenne
        train_loss /= len(train_loader)
        
        # Mode évaluation
        model.eval()
        val_loss = 0.0
        metrics.reset()
        
        with torch.no_grad():
            for batch in val_loader:
                # Extraire les données
                if args.model_type == "regression_unet_threshold":
                    inputs, thresholds, targets = [t.to(device) for t in batch]
                else:
                    inputs, targets = [t.to(device) for t in batch]
                    thresholds = None
                
                # Forward pass
                outputs = model(inputs, thresholds)
                
                # Calculer la perte
                loss = criterion(outputs, targets, thresholds)
                val_loss += loss.item()
                
                # Mettre à jour les métriques
                metrics.update(outputs, targets, thresholds)
        
        # Calculer la perte moyenne et les métriques
        val_loss /= len(val_loader)
        val_metrics = metrics.compute()
        
        # Mise à jour du scheduler
        scheduler.step(val_loss)
        
        # Journaliser dans TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        for metric_name, metric_value in val_metrics.items():
            if not metric_name.startswith('rmse_threshold_'):  # Éviter de surcharger TensorBoard
                writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
        
        # Afficher les résultats de l'époque
        logger.info(f"Époque {epoch+1}/{args.epochs}, "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"RMSE: {val_metrics['rmse']:.4f}, "
                   f"MAE: {val_metrics['mae']:.4f}, "
                   f"R²: {val_metrics['r2']:.4f}")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoints_dir / f"{args.model_type}_best.pt")
            logger.info(f"Meilleur modèle sauvegardé avec Val Loss: {val_loss:.4f}")
        
        # Sauvegarder le checkpoint de l'époque
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics
        }, checkpoints_dir / f"{args.model_type}_epoch_{epoch+1}.pt")
    
    # Évaluer le modèle sur l'ensemble de test
    logger.info("Évaluation du modèle sur l'ensemble de test...")
    
    # Charger le meilleur modèle
    model.load_state_dict(torch.load(checkpoints_dir / f"{args.model_type}_best.pt"))
    model.eval()
    
    # Réinitialiser les métriques
    metrics.reset()
    test_loss = 0.0
    
    # Stocker quelques exemples pour la visualisation
    examples = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Extraire les données
            if args.model_type == "regression_unet_threshold":
                inputs, thresholds, targets = [t.to(device) for t in batch]
            else:
                inputs, targets = [t.to(device) for t in batch]
                thresholds = None
            
            # Forward pass
            outputs = model(inputs, thresholds)
            
            # Calculer la perte
            loss = criterion(outputs, targets, thresholds)
            test_loss += loss.item()
            
            # Mettre à jour les métriques
            metrics.update(outputs, targets, thresholds)
            
            # Stocker quelques exemples pour la visualisation
            if i < 3:  # Sauvegarder les 3 premiers batchs
                for j in range(min(2, inputs.size(0))):  # 2 exemples par batch
                    examples.append({
                        'input': inputs[j].cpu().numpy(),
                        'target': targets[j].cpu().numpy(),
                        'prediction': outputs[j].cpu().numpy(),
                        'threshold': thresholds[j].cpu().numpy() if thresholds is not None else None
                    })
    
    # Calculer la perte et les métriques moyennes
    test_loss /= len(test_loader)
    test_metrics = metrics.compute()
    
    # Afficher les résultats
    logger.info(f"Résultats sur l'ensemble de test:")
    logger.info(f"Loss: {test_loss:.4f}")
    logger.info(f"RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"MAE: {test_metrics['mae']:.4f}")
    logger.info(f"R²: {test_metrics['r2']:.4f}")
    
    # Sauvegarder les métriques
    np.save(results_dir / "test_metrics.npy", test_metrics)
    
    # Visualiser quelques exemples
    for i, example in enumerate(examples):
        plot_regression_results(
            input_dsm=example['input'],
            target_chm=example['target'],
            predicted_chm=example['prediction'],
            save_path=results_dir / f"example_{i+1}.png"
        )
    
    # Fermer le writer TensorBoard
    writer.close()
    
    logger.info(f"Entraînement terminé. Résultats disponibles dans {output_dir}")


if __name__ == "__main__":
    main() 