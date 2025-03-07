"""
Classes de base pour le module d'évaluation externe.

Ce module contient les classes principales pour évaluer des modèles
sur des paires DSM/CHM indépendantes et générer des rapports d'évaluation.
"""

import os
import time
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import rasterio
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from forestgaps_dl.environment import setup_environment
from forestgaps_dl.config import ConfigurationManager, load_default_config
from forestgaps_dl.utils.io.serialization import load_model as load_model_checkpoint
from forestgaps_dl.models.registry import ModelRegistry

from forestgaps_dl.inference.utils.geospatial import load_raster, save_raster
from forestgaps_dl.inference.utils.processing import preprocess_dsm
from forestgaps_dl.inference.core import InferenceManager, InferenceResult

from .utils.metrics import calculate_metrics, calculate_threshold_metrics, calculate_confusion_matrix
from .utils.visualization import visualize_metrics, visualize_comparison, create_metrics_table
from .utils.reporting import generate_evaluation_report, save_metrics_to_csv

# Configuration du logging
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """
    Configuration pour l'évaluation externe.
    
    Attributes:
        thresholds: Liste des seuils de hauteur à évaluer
        compare_with_previous: Comparer avec une version précédente du modèle
        previous_model_path: Chemin vers le modèle précédent pour comparaison
        save_predictions: Sauvegarder les prédictions générées
        save_visualizations: Générer et sauvegarder des visualisations
        detailed_reporting: Générer des rapports détaillés
        metrics: Liste des métriques à calculer
        batch_size: Taille des lots pour le traitement par lots
        num_workers: Nombre de processus parallèles pour le chargement des données
        tiled_processing: Utiliser le traitement par tuiles pour les grandes images
    """
    thresholds: List[float] = None
    compare_with_previous: bool = False
    previous_model_path: Optional[str] = None
    save_predictions: bool = True
    save_visualizations: bool = True
    detailed_reporting: bool = True
    metrics: List[str] = None
    batch_size: int = 1
    num_workers: int = 4
    tiled_processing: bool = False


class EvaluationResult:
    """
    Résultat d'une évaluation externe.
    
    Cette classe encapsule les résultats d'une évaluation, y compris
    les métriques, les prédictions, et les visualisations.
    """
    
    def __init__(
        self,
        metrics: Dict[str, Any],
        thresholds: List[float],
        predictions: Optional[Dict[float, np.ndarray]] = None,
        ground_truth: Optional[Dict[float, np.ndarray]] = None,
        input_dsm: Optional[np.ndarray] = None,
        input_chm: Optional[np.ndarray] = None,
        input_paths: Optional[Dict[str, str]] = None,
        output_paths: Optional[Dict[str, str]] = None,
        processing_time: Optional[float] = None,
        confusion_matrices: Optional[Dict[float, Dict[str, int]]] = None
    ):
        """
        Initialise un résultat d'évaluation.
        
        Args:
            metrics: Métriques d'évaluation par seuil et métriques moyennes
            thresholds: Liste des seuils de hauteur évalués
            predictions: Prédictions par seuil (optionnel)
            ground_truth: Vérités terrain par seuil (optionnel)
            input_dsm: DSM d'entrée (optionnel)
            input_chm: CHM d'entrée (optionnel)
            input_paths: Chemins des fichiers d'entrée (optionnel)
            output_paths: Chemins des fichiers de sortie (optionnel)
            processing_time: Temps de traitement en secondes (optionnel)
            confusion_matrices: Matrices de confusion par seuil (optionnel)
        """
        self.metrics = metrics
        self.thresholds = thresholds
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.input_dsm = input_dsm
        self.input_chm = input_chm
        self.input_paths = input_paths if input_paths is not None else {}
        self.output_paths = output_paths if output_paths is not None else {}
        self.processing_time = processing_time
        self.confusion_matrices = confusion_matrices
    
    def save_metrics(self, output_path: str) -> str:
        """
        Sauvegarde les métriques d'évaluation dans un fichier CSV.
        
        Args:
            output_path: Chemin où sauvegarder les métriques
            
        Returns:
            Chemin du fichier sauvegardé
        """
        return save_metrics_to_csv(self.metrics, output_path)
    
    def visualize(self, output_dir: str) -> str:
        """
        Génère des visualisations des résultats d'évaluation.
        
        Args:
            output_dir: Répertoire où sauvegarder les visualisations
            
        Returns:
            Chemin du répertoire de visualisations
        """
        # Créer le répertoire si nécessaire
        os.makedirs(output_dir, exist_ok=True)
        
        # Générer les visualisations de métriques
        visualize_metrics(
            metrics=self.metrics,
            thresholds=self.thresholds,
            output_dir=output_dir
        )
        
        # Générer les visualisations de comparaison si les prédictions sont disponibles
        if self.predictions is not None and self.ground_truth is not None:
            for threshold in self.thresholds:
                visualize_comparison(
                    prediction=self.predictions[threshold],
                    ground_truth=self.ground_truth[threshold],
                    dsm=self.input_dsm,
                    threshold=threshold,
                    output_dir=output_dir
                )
        
        # Créer une table de métriques
        create_metrics_table(
            metrics=self.metrics,
            output_path=os.path.join(output_dir, "metrics_table.png")
        )
        
        return output_dir
    
    def generate_report(self, output_path: str) -> str:
        """
        Génère un rapport d'évaluation complet.
        
        Args:
            output_path: Chemin où sauvegarder le rapport
            
        Returns:
            Chemin du rapport généré
        """
        return generate_evaluation_report(
            result=self,
            output_path=output_path
        )
    
    @classmethod
    def aggregate_results(cls, results: List['EvaluationResult']) -> 'EvaluationResult':
        """
        Agrège plusieurs résultats d'évaluation en un seul.
        
        Args:
            results: Liste des résultats d'évaluation à agréger
            
        Returns:
            Résultat d'évaluation agrégé
        """
        # Vérifier s'il y a des résultats à agréger
        if not results:
            raise ValueError("Aucun résultat à agréger")
        
        # Extraire les thresholds (supposés identiques pour tous les résultats)
        thresholds = results[0].thresholds
        
        # Initialiser les métriques agrégées
        aggregated_metrics = {
            "mean_iou": 0.0,
            "mean_f1": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "by_threshold": {t: {
                "iou": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0
            } for t in thresholds}
        }
        
        # Agréger les métriques
        for result in results:
            aggregated_metrics["mean_iou"] += result.metrics["mean_iou"]
            aggregated_metrics["mean_f1"] += result.metrics["mean_f1"]
            aggregated_metrics["mean_precision"] += result.metrics["mean_precision"]
            aggregated_metrics["mean_recall"] += result.metrics["mean_recall"]
            
            for threshold in thresholds:
                aggregated_metrics["by_threshold"][threshold]["iou"] += result.metrics["by_threshold"][threshold]["iou"]
                aggregated_metrics["by_threshold"][threshold]["f1"] += result.metrics["by_threshold"][threshold]["f1"]
                aggregated_metrics["by_threshold"][threshold]["precision"] += result.metrics["by_threshold"][threshold]["precision"]
                aggregated_metrics["by_threshold"][threshold]["recall"] += result.metrics["by_threshold"][threshold]["recall"]
        
        # Calculer les moyennes
        n = len(results)
        aggregated_metrics["mean_iou"] /= n
        aggregated_metrics["mean_f1"] /= n
        aggregated_metrics["mean_precision"] /= n
        aggregated_metrics["mean_recall"] /= n
        
        for threshold in thresholds:
            aggregated_metrics["by_threshold"][threshold]["iou"] /= n
            aggregated_metrics["by_threshold"][threshold]["f1"] /= n
            aggregated_metrics["by_threshold"][threshold]["precision"] /= n
            aggregated_metrics["by_threshold"][threshold]["recall"] /= n
        
        # Créer le résultat agrégé
        return cls(
            metrics=aggregated_metrics,
            thresholds=thresholds,
            processing_time=sum(r.processing_time for r in results if r.processing_time) / n if any(r.processing_time for r in results) else None
        )


class ExternalEvaluator:
    """
    Évaluateur pour des modèles sur des paires DSM/CHM indépendantes.
    
    Cette classe fournit des méthodes pour évaluer des modèles préentraînés
    sur des paires DSM/CHM indépendantes et générer des rapports d'évaluation.
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 4
    ):
        """
        Initialise l'évaluateur externe.
        
        Args:
            model_path: Chemin vers le modèle préentraîné (.pt)
            config: Configuration pour l'évaluation (optionnel)
            device: Dispositif sur lequel exécuter l'évaluation ('cpu', 'cuda', etc.)
            batch_size: Taille des lots pour le traitement par lots
            num_workers: Nombre de processus parallèles pour le chargement des données
        """
        self.model_path = model_path
        
        # Configurer l'environnement
        self.env = setup_environment()
        
        # Déterminer le dispositif
        if device is None:
            self.device = self.env.get_device()
        else:
            self.device = torch.device(device)
        
        # Charger la configuration
        self.config = EvaluationConfig()
        self.config.batch_size = batch_size
        self.config.num_workers = num_workers
        
        # Définir les métriques par défaut si nécessaire
        if self.config.metrics is None:
            self.config.metrics = ["iou", "f1", "precision", "recall"]
        
        # Définir les seuils par défaut si nécessaire
        if self.config.thresholds is None:
            self.config.thresholds = [2.0, 5.0, 10.0, 15.0]
        
        if config is not None:
            # Mettre à jour la configuration avec les valeurs fournies
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Créer le gestionnaire d'inférence
        self.inference_manager = InferenceManager(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            config={
                "tiled_processing": self.config.tiled_processing,
                "save_probability": True  # Toujours sauvegarder les probabilités pour l'évaluation
            }
        )
    
    def _create_ground_truth(self, chm_data: np.ndarray, thresholds: List[float]) -> Dict[float, np.ndarray]:
        """
        Crée les vérités terrain à partir des données CHM.
        
        Args:
            chm_data: Données CHM
            thresholds: Seuils de hauteur
            
        Returns:
            Dictionnaire des vérités terrain par seuil
        """
        ground_truth = {}
        
        for threshold in thresholds:
            # Créer le masque de vérité terrain (1 où CHM < threshold, 0 ailleurs)
            gt = (chm_data < threshold).astype(np.float32)
            ground_truth[threshold] = gt
        
        return ground_truth
    
    def evaluate(
        self,
        dsm_path: str,
        chm_path: str,
        thresholds: Optional[List[float]] = None,
        output_dir: Optional[str] = None,
        visualize: bool = False
    ) -> EvaluationResult:
        """
        Évalue un modèle sur une paire DSM/CHM.
        
        Args:
            dsm_path: Chemin vers le fichier DSM d'entrée
            chm_path: Chemin vers le fichier CHM pour la vérité terrain
            thresholds: Liste des seuils de hauteur à évaluer (optionnel)
            output_dir: Répertoire pour sauvegarder les résultats (optionnel)
            visualize: Générer des visualisations des résultats
            
        Returns:
            Résultat de l'évaluation
        """
        logger.info(f"Évaluation du modèle sur la paire DSM/CHM: {dsm_path} / {chm_path}")
        
        start_time = time.time()
        
        # Utiliser les seuils par défaut si non spécifiés
        if thresholds is None:
            thresholds = self.config.thresholds
        
        # Créer le répertoire de sortie si spécifié
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Charger les données DSM et CHM
        dsm_data, dsm_metadata = load_raster(dsm_path)
        chm_data, chm_metadata = load_raster(chm_path)
        
        # Vérifier que les dimensions correspondent
        if dsm_data.shape != chm_data.shape:
            raise ValueError(f"Les dimensions DSM {dsm_data.shape} et CHM {chm_data.shape} ne correspondent pas")
        
        # Créer les vérités terrain pour chaque seuil
        ground_truth = self._create_ground_truth(chm_data, thresholds)
        
        # Dictionnaire pour stocker les prédictions par seuil
        predictions = {}
        
        # Exécuter l'inférence pour chaque seuil
        for threshold in thresholds:
            # Définir le chemin de sortie pour cette prédiction
            if output_dir is not None and self.config.save_predictions:
                output_path = os.path.join(output_dir, f"prediction_{threshold:.1f}.tif")
            else:
                output_path = None
            
            # Exécuter l'inférence
            inference_result = self.inference_manager.predict(
                dsm_path=dsm_path,
                threshold=threshold,
                output_path=output_path,
                visualize=False  # Nous générerons nos propres visualisations d'évaluation
            )
            
            # Stocker la prédiction
            predictions[threshold] = inference_result.prediction
        
        # Calculer les métriques
        metrics = {}
        confusion_matrices = {}
        
        # Métriques moyennes
        mean_iou = 0.0
        mean_f1 = 0.0
        mean_precision = 0.0
        mean_recall = 0.0
        
        # Calculer les métriques pour chaque seuil
        metrics["by_threshold"] = {}
        
        for threshold in thresholds:
            # Calculer les métriques pour ce seuil
            threshold_metrics = calculate_threshold_metrics(
                prediction=predictions[threshold],
                ground_truth=ground_truth[threshold]
            )
            
            # Stocker les métriques
            metrics["by_threshold"][threshold] = threshold_metrics
            
            # Accumuler pour les moyennes
            mean_iou += threshold_metrics["iou"]
            mean_f1 += threshold_metrics["f1"]
            mean_precision += threshold_metrics["precision"]
            mean_recall += threshold_metrics["recall"]
            
            # Calculer la matrice de confusion
            confusion_matrices[threshold] = calculate_confusion_matrix(
                prediction=predictions[threshold],
                ground_truth=ground_truth[threshold]
            )
        
        # Calculer les moyennes
        metrics["mean_iou"] = mean_iou / len(thresholds)
        metrics["mean_f1"] = mean_f1 / len(thresholds)
        metrics["mean_precision"] = mean_precision / len(thresholds)
        metrics["mean_recall"] = mean_recall / len(thresholds)
        
        # Calculer le temps de traitement
        processing_time = time.time() - start_time
        
        # Créer le résultat
        result = EvaluationResult(
            metrics=metrics,
            thresholds=thresholds,
            predictions=predictions if self.config.save_predictions else None,
            ground_truth=ground_truth,
            input_dsm=dsm_data,
            input_chm=chm_data,
            input_paths={"dsm": dsm_path, "chm": chm_path},
            processing_time=processing_time,
            confusion_matrices=confusion_matrices
        )
        
        # Sauvegarder les métriques si un répertoire de sortie est spécifié
        if output_dir is not None:
            metrics_path = os.path.join(output_dir, "metrics.csv")
            result.save_metrics(metrics_path)
            logger.info(f"Métriques sauvegardées dans: {metrics_path}")
        
        # Générer des visualisations si demandé
        if visualize and output_dir is not None:
            vis_dir = os.path.join(output_dir, "visualizations")
            result.visualize(vis_dir)
            logger.info(f"Visualisations générées dans: {vis_dir}")
            
            # Générer un rapport si demandé
            if self.config.detailed_reporting:
                report_path = os.path.join(output_dir, "evaluation_report.html")
                result.generate_report(report_path)
                logger.info(f"Rapport d'évaluation généré: {report_path}")
        
        logger.info(f"Évaluation terminée en {processing_time:.2f} secondes")
        logger.info(f"IoU moyen: {metrics['mean_iou']:.4f}, F1 moyen: {metrics['mean_f1']:.4f}")
        
        return result
    
    def evaluate_site(
        self,
        site_dsm_dir: str,
        site_chm_dir: str,
        thresholds: Optional[List[float]] = None,
        output_dir: Optional[str] = None,
        visualize: bool = False
    ) -> EvaluationResult:
        """
        Évalue un modèle sur un site complet (plusieurs paires DSM/CHM).
        
        Args:
            site_dsm_dir: Répertoire contenant les fichiers DSM du site
            site_chm_dir: Répertoire contenant les fichiers CHM du site
            thresholds: Liste des seuils de hauteur à évaluer (optionnel)
            output_dir: Répertoire pour sauvegarder les résultats (optionnel)
            visualize: Générer des visualisations des résultats
            
        Returns:
            Résultat d'évaluation agrégé pour tout le site
        """
        logger.info(f"Évaluation du modèle sur le site: {site_dsm_dir}")
        
        # Utiliser les seuils par défaut si non spécifiés
        if thresholds is None:
            thresholds = self.config.thresholds
        
        # Créer le répertoire de sortie si spécifié
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Trouver toutes les paires DSM/CHM dans les répertoires
        dsm_files = [f for f in os.listdir(site_dsm_dir) if f.endswith(('.tif', '.tiff'))]
        chm_files = [f for f in os.listdir(site_chm_dir) if f.endswith(('.tif', '.tiff'))]
        
        # Associer les fichiers DSM et CHM par nom (supposé même nom de base)
        pairs = []
        
        for dsm_file in dsm_files:
            dsm_base = os.path.splitext(dsm_file)[0]
            
            # Chercher le fichier CHM correspondant
            chm_match = None
            for chm_file in chm_files:
                chm_base = os.path.splitext(chm_file)[0]
                if dsm_base == chm_base or dsm_base in chm_base or chm_base in dsm_base:
                    chm_match = chm_file
                    break
            
            if chm_match:
                pairs.append((
                    os.path.join(site_dsm_dir, dsm_file),
                    os.path.join(site_chm_dir, chm_match)
                ))
        
        if not pairs:
            raise ValueError("Aucune paire DSM/CHM correspondante trouvée dans les répertoires spécifiés")
        
        logger.info(f"Trouvé {len(pairs)} paires DSM/CHM à évaluer")
        
        # Évaluer chaque paire
        results = []
        
        for dsm_path, chm_path in tqdm(pairs, desc="Évaluation des paires DSM/CHM"):
            # Créer un sous-répertoire pour cette paire si un répertoire de sortie est spécifié
            if output_dir is not None:
                pair_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(dsm_path))[0])
                os.makedirs(pair_dir, exist_ok=True)
            else:
                pair_dir = None
            
            # Évaluer la paire
            try:
                result = self.evaluate(
                    dsm_path=dsm_path,
                    chm_path=chm_path,
                    thresholds=thresholds,
                    output_dir=pair_dir,
                    visualize=visualize
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Erreur lors de l'évaluation de la paire {dsm_path}/{chm_path}: {str(e)}")
                continue
        
        # Agréger les résultats
        aggregated_result = EvaluationResult.aggregate_results(results)
        
        # Sauvegarder les métriques agrégées si un répertoire de sortie est spécifié
        if output_dir is not None:
            metrics_path = os.path.join(output_dir, "site_metrics.csv")
            aggregated_result.save_metrics(metrics_path)
            logger.info(f"Métriques du site sauvegardées dans: {metrics_path}")
            
            # Générer un rapport si demandé
            if self.config.detailed_reporting:
                report_path = os.path.join(output_dir, "site_evaluation_report.html")
                aggregated_result.generate_report(report_path)
                logger.info(f"Rapport d'évaluation du site généré: {report_path}")
        
        logger.info(f"Évaluation du site terminée, {len(results)} paires traitées avec succès")
        logger.info(f"IoU moyen du site: {aggregated_result.metrics['mean_iou']:.4f}, F1 moyen: {aggregated_result.metrics['mean_f1']:.4f}")
        
        return aggregated_result
    
    def evaluate_multi_sites(
        self,
        sites_config: Dict[str, Dict[str, str]],
        thresholds: Optional[List[float]] = None,
        output_dir: Optional[str] = None,
        visualize: bool = False,
        aggregate_results: bool = True
    ) -> Dict[str, EvaluationResult]:
        """
        Évalue un modèle sur plusieurs sites.
        
        Args:
            sites_config: Configuration des sites {nom_site: {"dsm_dir": path, "chm_dir": path}}
            thresholds: Liste des seuils de hauteur à évaluer (optionnel)
            output_dir: Répertoire pour sauvegarder les résultats (optionnel)
            visualize: Générer des visualisations des résultats
            aggregate_results: Agréger les résultats de tous les sites
            
        Returns:
            Dictionnaire des résultats d'évaluation pour chaque site et résultat agrégé si demandé
        """
        logger.info(f"Évaluation du modèle sur {len(sites_config)} sites")
        
        # Utiliser les seuils par défaut si non spécifiés
        if thresholds is None:
            thresholds = self.config.thresholds
        
        # Créer le répertoire de sortie si spécifié
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Résultats pour chaque site
        results = {}
        all_site_results = []
        
        # Évaluer chaque site
        for site_name, site_dirs in sites_config.items():
            logger.info(f"Évaluation du site: {site_name}")
            
            # Extraire les répertoires DSM et CHM
            dsm_dir = site_dirs.get("dsm_dir")
            chm_dir = site_dirs.get("chm_dir")
            
            if not dsm_dir or not chm_dir:
                logger.warning(f"Configuration incomplète pour le site {site_name}, ignoré")
                continue
            
            # Créer un sous-répertoire pour ce site si un répertoire de sortie est spécifié
            if output_dir is not None:
                site_dir = os.path.join(output_dir, site_name)
                os.makedirs(site_dir, exist_ok=True)
            else:
                site_dir = None
            
            # Évaluer le site
            try:
                site_result = self.evaluate_site(
                    site_dsm_dir=dsm_dir,
                    site_chm_dir=chm_dir,
                    thresholds=thresholds,
                    output_dir=site_dir,
                    visualize=visualize
                )
                
                results[site_name] = site_result
                all_site_results.append(site_result)
                
            except Exception as e:
                logger.error(f"Erreur lors de l'évaluation du site {site_name}: {str(e)}")
                continue
        
        # Agréger les résultats de tous les sites si demandé
        if aggregate_results and all_site_results:
            logger.info("Agrégation des résultats de tous les sites")
            
            aggregated_result = EvaluationResult.aggregate_results(all_site_results)
            results["aggregated"] = aggregated_result
            
            # Sauvegarder les métriques agrégées si un répertoire de sortie est spécifié
            if output_dir is not None:
                metrics_path = os.path.join(output_dir, "multi_sites_metrics.csv")
                aggregated_result.save_metrics(metrics_path)
                logger.info(f"Métriques agrégées sauvegardées dans: {metrics_path}")
                
                # Générer un rapport si demandé
                if self.config.detailed_reporting:
                    report_path = os.path.join(output_dir, "multi_sites_evaluation_report.html")
                    aggregated_result.generate_report(report_path)
                    logger.info(f"Rapport d'évaluation multi-sites généré: {report_path}")
            
            logger.info(f"IoU moyen global: {aggregated_result.metrics['mean_iou']:.4f}, F1 moyen: {aggregated_result.metrics['mean_f1']:.4f}")
        
        logger.info(f"Évaluation multi-sites terminée, {len(results)-1 if 'aggregated' in results else len(results)} sites traités avec succès")
        
        return results 