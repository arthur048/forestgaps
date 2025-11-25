"""
Module de comparaison de modèles pour ForestGaps.

Ce module fournit la classe principale pour gérer la comparaison
systématique de différents modèles et configurations.
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader

from forestgaps.config import Config
from forestgaps.models import ModelRegistry
from forestgaps.training import Trainer
from forestgaps.training.metrics import SegmentationMetrics
from forestgaps.environment import get_device
from forestgaps.utils.io.serialization import save_json, load_json
from forestgaps.utils.errors import BenchmarkingError

from benchmarking.metrics import AggregatedMetrics, MetricsTracker
from benchmarking.visualization import BenchmarkVisualizer
from benchmarking.reporting import BenchmarkReporter


logger = logging.getLogger(__name__)


class ModelComparison:
    """
    Classe principale pour la comparaison de modèles ForestGaps.
    
    Cette classe permet de comparer systématiquement différentes architectures
    de modèles et configurations d'entraînement sur les mêmes données,
    en collectant des métriques détaillées et en générant des rapports
    et visualisations comparatives.
    """
    
    def __init__(
        self, 
        model_configs: List[Dict[str, Any]],
        base_config: Config,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        metrics: Optional[List[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        threshold_values: Optional[List[float]] = None
    ):
        """
        Initialise une comparaison de modèles.
        
        Args:
            model_configs: Liste de configurations de modèles à comparer.
                Chaque élément doit contenir au moins:
                - 'name': Nom du modèle (correspond au registre)
                - 'params': Paramètres spécifiques du modèle
            base_config: Configuration de base commune à tous les modèles.
            train_loader: DataLoader pour l'entraînement (si None, sera créé à partir de base_config).
            val_loader: DataLoader pour la validation (si None, sera créé à partir de base_config).
            test_loader: DataLoader pour le test (si None, sera créé à partir de base_config).
            metrics: Liste des métriques à collecter (par défaut: ['iou', 'dice', 'accuracy']).
            output_dir: Répertoire de sortie pour les résultats.
            device: Dispositif à utiliser ('cuda' ou 'cpu', détection auto si None).
            threshold_values: Valeurs de seuil à utiliser pour les métriques par seuil.
        """
        self.model_configs = model_configs
        self.base_config = base_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metrics = metrics or ['iou', 'dice', 'accuracy']
        self.device = device or get_device()
        self.threshold_values = threshold_values or [2.0, 5.0, 10.0, 15.0]
        
        # Configurer le répertoire de sortie
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir or f"benchmark_results_{timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialiser les trackers de métriques
        self.metrics_trackers = {}
        self.results = {}
        self.best_models = {}
        
        # Vérifier les configurations des modèles
        self._validate_model_configs()
        
        # Journalisation
        logger.info(f"Initialisation de la comparaison de modèles avec {len(model_configs)} modèles")
        logger.info(f"Métriques suivies: {self.metrics}")
        logger.info(f"Valeurs de seuil: {self.threshold_values}")
        logger.info(f"Répertoire de sortie: {self.output_dir}")
    
    def _validate_model_configs(self) -> None:
        """
        Vérifie que les configurations des modèles sont valides.
        """
        for i, config in enumerate(self.model_configs):
            if 'name' not in config:
                raise BenchmarkingError(f"La configuration du modèle {i} ne contient pas de 'name'")
            
            model_type = config['name']
            if not ModelRegistry.get_model_class(model_type):
                available_models = ModelRegistry.list_models()
                raise BenchmarkingError(
                    f"Le modèle '{model_type}' n'est pas enregistré. "
                    f"Options disponibles: {available_models}"
                )
    
    def _prepare_data_loaders(self) -> None:
        """
        Prépare les data loaders s'ils n'ont pas été fournis.
        """
        if self.train_loader is None or self.val_loader is None:
            from forestgaps.data.loaders import create_data_loaders
            
            logger.info("Création des data loaders à partir de la configuration")
            loaders = create_data_loaders(self.base_config)
            
            self.train_loader = loaders.get('train')
            self.val_loader = loaders.get('val')
            self.test_loader = loaders.get('test')
    
    def _create_trainer(self, model_config: Dict[str, Any]) -> Tuple[Trainer, torch.nn.Module]:
        """
        Crée un Trainer et un modèle à partir d'une configuration.
        
        Args:
            model_config: Configuration du modèle.
            
        Returns:
            Tuple contenant le Trainer et le modèle.
        """
        model_type = model_config['name']
        model_params = model_config.get('params', {})
        
        # Créer le modèle
        model = ModelRegistry.create(model_type, **model_params)
        model.to(self.device)
        
        # Créer le trainer
        trainer = Trainer(
            model=model,
            config=self.base_config,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            device=self.device
        )
        
        return trainer, model
    
    def run(self) -> Dict[str, Any]:
        """
        Exécute la comparaison de modèles.
        
        Returns:
            Dictionnaire contenant les résultats de la comparaison.
        """
        logger.info("Début de la comparaison des modèles")
        
        # Préparer les data loaders si nécessaire
        self._prepare_data_loaders()
        
        # Exécuter l'entraînement pour chaque modèle
        for i, model_config in enumerate(self.model_configs):
            model_type = model_config['name']
            model_name = model_config.get('display_name', model_type)
            
            logger.info(f"[{i+1}/{len(self.model_configs)}] Entraînement du modèle {model_name}")
            
            # Créer le répertoire pour ce modèle
            model_dir = self.output_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Créer le trainer et le modèle
            trainer, model = self._create_trainer(model_config)
            
            # Enregistrer la configuration
            config_path = model_dir / "model_config.json"
            save_json(model_config, config_path)
            
            # Entraîner le modèle et mesurer le temps
            start_time = time.time()
            train_results = trainer.train(
                epochs=self.base_config.get('training', {}).get('epochs', 50),
                log_dir=model_dir,
                save_checkpoints=True
            )
            training_time = time.time() - start_time
            
            # Évaluer le modèle
            test_results = trainer.evaluate(
                data_loader=self.test_loader,
                threshold_values=self.threshold_values
            )
            
            # Collecter les métriques
            metrics_tracker = MetricsTracker(
                model_name=model_name,
                metrics=self.metrics,
                threshold_values=self.threshold_values
            )
            
            metrics_tracker.update(
                train_metrics=train_results.get('metrics', {}),
                val_metrics=train_results.get('val_metrics', {}),
                test_metrics=test_results,
                training_time=training_time,
                model_params=model_config.get('params', {})
            )
            
            self.metrics_trackers[model_name] = metrics_tracker
            
            # Enregistrer les métriques
            metrics_path = model_dir / "metrics.json"
            save_json(metrics_tracker.get_data(), metrics_path)
            
            # Enregistrer les prédictions pour quelques exemples
            if self.test_loader is not None:
                self._save_prediction_examples(
                    model=model,
                    data_loader=self.test_loader,
                    output_dir=model_dir,
                    num_examples=5
                )
        
        # Agréger les résultats
        self.results = self._aggregate_results()
        
        # Enregistrer les résultats complets
        results_path = self.output_dir / "benchmark_results.json"
        save_json(self.results, results_path)
        
        # Générer les visualisations
        self._generate_visualizations()
        
        # Générer le rapport
        self._generate_report()
        
        logger.info(f"Comparaison de modèles terminée. Résultats disponibles dans {self.output_dir}")
        
        return self.results
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """
        Agrège les résultats de tous les modèles.
        
        Returns:
            Dictionnaire contenant les résultats agrégés.
        """
        aggregated_metrics = AggregatedMetrics(
            metrics_trackers=list(self.metrics_trackers.values()),
            metrics=self.metrics,
            threshold_values=self.threshold_values
        )
        
        results = {
            'summary': aggregated_metrics.get_summary(),
            'best_models': aggregated_metrics.get_best_models(),
            'models': {
                name: tracker.get_data() 
                for name, tracker in self.metrics_trackers.items()
            },
            'config': {
                'base_config': {k: v for k, v in self.base_config.items() if isinstance(v, (str, int, float, bool, list, dict))},
                'metrics': self.metrics,
                'threshold_values': self.threshold_values
            },
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self.best_models = results['best_models']
        return results
    
    def _save_prediction_examples(
        self, 
        model: torch.nn.Module, 
        data_loader: DataLoader, 
        output_dir: Path,
        num_examples: int = 5
    ) -> None:
        """
        Sauvegarde quelques exemples de prédictions pour visualisation.
        
        Args:
            model: Modèle entraîné
            data_loader: DataLoader contenant les données
            output_dir: Répertoire où sauvegarder les exemples
            num_examples: Nombre d'exemples à sauvegarder
        """
        examples_dir = output_dir / "prediction_examples"
        examples_dir.mkdir(exist_ok=True)
        
        # Mettre le modèle en mode évaluation
        model.eval()
        
        # Sélectionner quelques exemples
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_examples:
                    break
                
                # Extraire les données
                if isinstance(batch, dict):
                    inputs = batch['dsm'].to(self.device)
                    thresholds = batch['threshold'].to(self.device)
                    targets = batch['mask'].to(self.device)
                else:
                    inputs, thresholds, targets = [t.to(self.device) for t in batch]
                
                # Faire la prédiction
                outputs = model(inputs, thresholds)
                
                # Sauvegarder les entrées, cibles et prédictions
                example_data = {
                    'inputs': inputs.cpu().numpy(),
                    'thresholds': thresholds.cpu().numpy(),
                    'targets': targets.cpu().numpy(),
                    'predictions': outputs.cpu().numpy()
                }
                
                # Sauvegarder en format numpy
                np.save(examples_dir / f"example_{i}.npy", example_data)
    
    def _generate_visualizations(self) -> None:
        """
        Génère les visualisations comparatives.
        """
        visualizer = BenchmarkVisualizer(
            results=self.results,
            output_dir=self.output_dir
        )
        
        visualizer.generate_all()
    
    def _generate_report(self) -> None:
        """
        Génère un rapport détaillé de la comparaison.
        """
        reporter = BenchmarkReporter(
            results=self.results,
            output_dir=self.output_dir
        )
        
        reporter.generate_report()
    
    def visualize_results(self) -> None:
        """
        Génère et affiche les visualisations des résultats.
        """
        if not self.results:
            logger.warning("Aucun résultat disponible. Exécutez d'abord la méthode 'run()'.")
            return
        
        visualizer = BenchmarkVisualizer(
            results=self.results,
            output_dir=self.output_dir
        )
        
        visualizer.generate_all(show=True)
    
    def get_best_model(self, metric: str = 'iou', threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Retourne la configuration du meilleur modèle selon une métrique.
        
        Args:
            metric: Métrique à utiliser pour la comparaison ('iou', 'dice', etc.)
            threshold: Seuil de hauteur spécifique (si None, utilise la moyenne)
            
        Returns:
            Configuration du meilleur modèle
        """
        if not self.best_models:
            logger.warning("Aucun résultat disponible. Exécutez d'abord la méthode 'run()'.")
            return {}
        
        if threshold is not None:
            best_key = f"{metric}_threshold_{threshold}"
        else:
            best_key = f"{metric}_average"
        
        if best_key not in self.best_models:
            logger.warning(f"Métrique '{best_key}' non disponible. Utilisation de 'iou_average'.")
            best_key = "iou_average"
        
        best_model_name = self.best_models.get(best_key)
        
        if not best_model_name:
            return {}
        
        # Trouver la configuration correspondante
        for config in self.model_configs:
            if config.get('display_name', config['name']) == best_model_name:
                return config
        
        return {}
    
    def save_best_model(self, output_path: Union[str, Path], metric: str = 'iou', threshold: Optional[float] = None) -> Optional[Path]:
        """
        Sauvegarde le meilleur modèle selon une métrique.
        
        Args:
            output_path: Chemin où sauvegarder le modèle
            metric: Métrique à utiliser pour la comparaison ('iou', 'dice', etc.)
            threshold: Seuil de hauteur spécifique (si None, utilise la moyenne)
            
        Returns:
            Chemin du modèle sauvegardé ou None si échec
        """
        best_config = self.get_best_model(metric, threshold)
        
        if not best_config:
            logger.warning("Impossible de trouver le meilleur modèle.")
            return None
        
        model_name = best_config.get('display_name', best_config['name'])
        model_dir = self.output_dir / model_name
        
        # Chercher le dernier checkpoint
        checkpoints_dir = model_dir / "checkpoints"
        if not checkpoints_dir.exists():
            logger.warning(f"Répertoire de checkpoints non trouvé pour {model_name}.")
            return None
        
        checkpoints = list(checkpoints_dir.glob("*.pt"))
        if not checkpoints:
            logger.warning(f"Aucun checkpoint trouvé pour {model_name}.")
            return None
        
        # Trouver le meilleur checkpoint
        best_checkpoint = None
        best_val = -float('inf')
        
        for checkpoint in checkpoints:
            if "best" in checkpoint.name:
                best_checkpoint = checkpoint
                break
            
            # Extraire le numéro d'époque
            try:
                epoch = int(checkpoint.stem.split('_')[-1])
                if epoch > best_val:
                    best_val = epoch
                    best_checkpoint = checkpoint
            except (ValueError, IndexError):
                continue
        
        if best_checkpoint is None:
            logger.warning(f"Impossible de déterminer le meilleur checkpoint pour {model_name}.")
            return None
        
        # Copier le checkpoint vers la destination
        import shutil
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy(best_checkpoint, output_path)
        logger.info(f"Meilleur modèle ({model_name}) sauvegardé dans {output_path}")
        
        return output_path 