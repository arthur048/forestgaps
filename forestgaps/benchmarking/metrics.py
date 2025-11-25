"""
Module de suivi et d'agrégation des métriques pour ForestGaps.

Ce module fournit des classes pour collecter, suivre et agréger les métriques
de performance des modèles lors des comparaisons.
"""

import logging
from typing import Dict, List, Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Classe pour suivre les métriques d'un modèle spécifique.
    
    Cette classe collecte les métriques d'entraînement, de validation et de test
    pour un modèle donné, et fournit des méthodes pour les analyser.
    """
    
    def __init__(
        self, 
        model_name: str,
        metrics: List[str] = None,
        threshold_values: List[float] = None
    ):
        """
        Initialise un tracker de métriques pour un modèle.
        
        Args:
            model_name: Nom du modèle.
            metrics: Liste des métriques à suivre.
            threshold_values: Liste des valeurs de seuil pour lesquelles 
                              collecter des métriques.
        """
        self.model_name = model_name
        self.metrics = metrics or ['iou', 'dice', 'accuracy']
        self.threshold_values = threshold_values or [2.0, 5.0, 10.0, 15.0]
        
        # Stockage des métriques
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        self.training_time = 0.0
        self.model_params = {}
        
        # Historique d'entraînement
        self.history = {metric: [] for metric in self.metrics}
        self.val_history = {metric: [] for metric in self.metrics}
        
        logger.debug(f"Tracker de métriques initialisé pour le modèle '{model_name}'")
    
    def update(
        self,
        train_metrics: Optional[Dict[str, Any]] = None,
        val_metrics: Optional[Dict[str, Any]] = None,
        test_metrics: Optional[Dict[str, Any]] = None,
        training_time: Optional[float] = None,
        model_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Met à jour les métriques du modèle.
        
        Args:
            train_metrics: Métriques d'entraînement.
            val_metrics: Métriques de validation.
            test_metrics: Métriques de test.
            training_time: Temps d'entraînement en secondes.
            model_params: Paramètres du modèle.
        """
        if train_metrics:
            self.train_metrics = train_metrics
            
            # Mise à jour de l'historique
            for metric in self.metrics:
                if metric in train_metrics:
                    self.history[metric].append(train_metrics[metric])
        
        if val_metrics:
            self.val_metrics = val_metrics
            
            # Mise à jour de l'historique de validation
            for metric in self.metrics:
                if metric in val_metrics:
                    self.val_history[metric].append(val_metrics[metric])
        
        if test_metrics:
            self.test_metrics = test_metrics
        
        if training_time is not None:
            self.training_time = training_time
        
        if model_params:
            self.model_params = model_params
    
    def get_data(self) -> Dict[str, Any]:
        """
        Retourne toutes les données collectées.
        
        Returns:
            Dictionnaire contenant toutes les métriques et informations collectées.
        """
        return {
            'model_name': self.model_name,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'test_metrics': self.test_metrics,
            'training_time': self.training_time,
            'model_params': self.model_params,
            'history': self.history,
            'val_history': self.val_history
        }
    
    def get_best_epoch(self, metric: str = 'iou') -> int:
        """
        Retourne l'époque avec la meilleure valeur pour une métrique donnée.
        
        Args:
            metric: Métrique à utiliser ('iou', 'dice', etc.).
            
        Returns:
            Numéro de l'époque avec la meilleure valeur.
        """
        if not self.val_history.get(metric):
            return 0
        
        return np.argmax(self.val_history[metric])
    
    def get_convergence_speed(self, metric: str = 'iou', threshold: float = 0.9) -> int:
        """
        Calcule la vitesse de convergence en comptant le nombre d'époques
        nécessaires pour atteindre un certain pourcentage de la meilleure valeur.
        
        Args:
            metric: Métrique à utiliser ('iou', 'dice', etc.).
            threshold: Seuil en pourcentage de la meilleure valeur (0.9 = 90%).
            
        Returns:
            Nombre d'époques pour atteindre le seuil.
        """
        if not self.val_history.get(metric):
            return -1
        
        best_value = max(self.val_history[metric])
        threshold_value = threshold * best_value
        
        for i, value in enumerate(self.val_history[metric]):
            if value >= threshold_value:
                return i + 1
        
        return len(self.val_history[metric])
    
    def get_metric_by_threshold(self, metric: str = 'iou') -> Dict[float, float]:
        """
        Retourne les valeurs d'une métrique pour chaque seuil.
        
        Args:
            metric: Métrique à utiliser ('iou', 'dice', etc.).
            
        Returns:
            Dictionnaire avec les valeurs de seuil comme clés et les valeurs de métrique.
        """
        result = {}
        
        for threshold in self.threshold_values:
            key = f"{metric}_threshold_{threshold}"
            
            if key in self.test_metrics:
                result[threshold] = self.test_metrics[key]
        
        return result


class AggregatedMetrics:
    """
    Classe pour agréger les métriques de plusieurs modèles.
    
    Cette classe combine les métriques de plusieurs trackers et fournit
    des méthodes pour comparer les modèles selon différentes métriques.
    """
    
    def __init__(
        self, 
        metrics_trackers: List[MetricsTracker],
        metrics: List[str] = None,
        threshold_values: List[float] = None
    ):
        """
        Initialise un agrégateur de métriques.
        
        Args:
            metrics_trackers: Liste des trackers de métriques à agréger.
            metrics: Liste des métriques à analyser.
            threshold_values: Liste des valeurs de seuil à considérer.
        """
        self.trackers = metrics_trackers
        self.metrics = metrics or ['iou', 'dice', 'accuracy']
        self.threshold_values = threshold_values or [2.0, 5.0, 10.0, 15.0]
        
        logger.debug(f"Agrégateur de métriques initialisé avec {len(metrics_trackers)} modèles")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Génère un résumé des métriques pour tous les modèles.
        
        Returns:
            Dictionnaire contenant un résumé des métriques.
        """
        summary = {}
        
        # Résumé par modèle
        model_summaries = {}
        for tracker in self.trackers:
            model_name = tracker.model_name
            model_summary = {}
            
            # Métriques de test moyennes
            for metric in self.metrics:
                metric_values = []
                for threshold in self.threshold_values:
                    key = f"{metric}_threshold_{threshold}"
                    if key in tracker.test_metrics:
                        metric_values.append(tracker.test_metrics[key])
                
                if metric_values:
                    model_summary[f"{metric}_average"] = np.mean(metric_values)
                    model_summary[f"{metric}_std"] = np.std(metric_values)
                    model_summary[f"{metric}_min"] = np.min(metric_values)
                    model_summary[f"{metric}_max"] = np.max(metric_values)
            
            # Informations complémentaires
            model_summary['training_time'] = tracker.training_time
            model_summary['convergence_speed'] = {
                metric: tracker.get_convergence_speed(metric)
                for metric in self.metrics
            }
            
            model_summaries[model_name] = model_summary
        
        summary['models'] = model_summaries
        
        # Statistiques globales
        global_stats = {}
        for metric in self.metrics:
            metric_values = []
            for tracker in self.trackers:
                for threshold in self.threshold_values:
                    key = f"{metric}_threshold_{threshold}"
                    if key in tracker.test_metrics:
                        metric_values.append(tracker.test_metrics[key])
            
            if metric_values:
                global_stats[f"{metric}_global_average"] = np.mean(metric_values)
                global_stats[f"{metric}_global_std"] = np.std(metric_values)
                global_stats[f"{metric}_global_min"] = np.min(metric_values)
                global_stats[f"{metric}_global_max"] = np.max(metric_values)
        
        summary['global_stats'] = global_stats
        
        return summary
    
    def get_best_models(self) -> Dict[str, str]:
        """
        Détermine les meilleurs modèles selon différentes métriques.
        
        Returns:
            Dictionnaire avec les noms des métriques comme clés et les noms des modèles comme valeurs.
        """
        best_models = {}
        
        # Meilleur modèle par métrique (moyenne sur tous les seuils)
        for metric in self.metrics:
            best_value = -float('inf')
            best_model = None
            
            for tracker in self.trackers:
                model_name = tracker.model_name
                
                # Calculer la moyenne pour cette métrique sur tous les seuils
                metric_values = []
                for threshold in self.threshold_values:
                    key = f"{metric}_threshold_{threshold}"
                    if key in tracker.test_metrics:
                        metric_values.append(tracker.test_metrics[key])
                
                if metric_values:
                    avg_value = np.mean(metric_values)
                    if avg_value > best_value:
                        best_value = avg_value
                        best_model = model_name
            
            if best_model:
                best_models[f"{metric}_average"] = best_model
        
        # Meilleur modèle par métrique et par seuil
        for metric in self.metrics:
            for threshold in self.threshold_values:
                best_value = -float('inf')
                best_model = None
                
                for tracker in self.trackers:
                    model_name = tracker.model_name
                    key = f"{metric}_threshold_{threshold}"
                    
                    if key in tracker.test_metrics and tracker.test_metrics[key] > best_value:
                        best_value = tracker.test_metrics[key]
                        best_model = model_name
                
                if best_model:
                    best_models[f"{metric}_threshold_{threshold}"] = best_model
        
        # Meilleur modèle en termes de temps d'entraînement
        best_time = float('inf')
        best_model = None
        
        for tracker in self.trackers:
            if tracker.training_time < best_time:
                best_time = tracker.training_time
                best_model = tracker.model_name
        
        if best_model:
            best_models['training_time'] = best_model
        
        # Meilleur modèle en termes de vitesse de convergence
        for metric in self.metrics:
            best_speed = float('inf')
            best_model = None
            
            for tracker in self.trackers:
                speed = tracker.get_convergence_speed(metric)
                if speed > 0 and speed < best_speed:
                    best_speed = speed
                    best_model = tracker.model_name
            
            if best_model:
                best_models[f"convergence_{metric}"] = best_model
        
        return best_models
    
    def get_model_rankings(self, metric: str = 'iou') -> Dict[str, int]:
        """
        Génère un classement des modèles selon une métrique.
        
        Args:
            metric: Métrique à utiliser pour le classement.
            
        Returns:
            Dictionnaire avec les noms des modèles comme clés et leur rang comme valeurs.
        """
        # Calculer la valeur moyenne de la métrique pour chaque modèle
        model_values = []
        
        for tracker in self.trackers:
            metric_values = []
            for threshold in self.threshold_values:
                key = f"{metric}_threshold_{threshold}"
                if key in tracker.test_metrics:
                    metric_values.append(tracker.test_metrics[key])
            
            if metric_values:
                model_values.append((tracker.model_name, np.mean(metric_values)))
        
        # Trier par valeur décroissante
        model_values.sort(key=lambda x: x[1], reverse=True)
        
        # Créer le dictionnaire de classement
        rankings = {}
        for i, (model_name, _) in enumerate(model_values):
            rankings[model_name] = i + 1
        
        return rankings 