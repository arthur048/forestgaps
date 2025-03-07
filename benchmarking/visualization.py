"""
Module de visualisation pour les résultats de benchmarking.

Ce module fournit des outils pour générer des visualisations comparatives
des performances des différents modèles.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


class BenchmarkVisualizer:
    """
    Classe pour visualiser les résultats de benchmarking.
    
    Cette classe génère différentes visualisations à partir des résultats
    de benchmarking, notamment des graphiques de comparaison de métriques,
    des courbes d'apprentissage, etc.
    """
    
    def __init__(self, results: Dict[str, Any], output_dir: Union[str, Path]):
        """
        Initialise le visualiseur de benchmarking.
        
        Args:
            results: Résultats de benchmarking (sortie de ModelComparison.run()).
            output_dir: Répertoire où sauvegarder les visualisations.
        """
        self.results = results
        self.output_dir = Path(output_dir) / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extraire les configurations
        self.metrics = results.get('config', {}).get('metrics', ['iou', 'dice', 'accuracy'])
        self.threshold_values = results.get('config', {}).get('threshold_values', [2.0, 5.0, 10.0, 15.0])
        
        # Définir une palette de couleurs
        self.color_palette = plt.cm.tab10.colors
        
        logger.info(f"Initialisé BenchmarkVisualizer avec répertoire de sortie: {self.output_dir}")
    
    def generate_all(self, show: bool = False) -> None:
        """
        Génère toutes les visualisations disponibles.
        
        Args:
            show: Si True, affiche les visualisations en plus de les sauvegarder.
        """
        logger.info("Génération de toutes les visualisations")
        
        # Générer les visualisations
        self.plot_metric_comparison(show=show)
        self.plot_threshold_comparison(show=show)
        self.plot_training_curves(show=show)
        self.plot_training_time_comparison(show=show)
        self.plot_convergence_speed_comparison(show=show)
        self.plot_radar_chart(show=show)
        
        logger.info("Toutes les visualisations ont été générées")
    
    def plot_metric_comparison(self, metric: str = 'iou', show: bool = False) -> None:
        """
        Génère un graphique de comparaison des performances des modèles pour une métrique.
        
        Args:
            metric: Métrique à visualiser ('iou', 'dice', etc.).
            show: Si True, affiche le graphique en plus de le sauvegarder.
        """
        logger.debug(f"Génération du graphique de comparaison pour la métrique '{metric}'")
        
        # Récupérer les données
        model_names = []
        metric_values = []
        
        for model_name, model_data in self.results.get('models', {}).items():
            summary = self.results.get('summary', {}).get('models', {}).get(model_name, {})
            if f"{metric}_average" in summary:
                model_names.append(model_name)
                metric_values.append(summary[f"{metric}_average"])
        
        if not model_names:
            logger.warning(f"Aucune donnée disponible pour la métrique '{metric}'")
            return
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(model_names, metric_values, color=self.color_palette[:len(model_names)])
        
        # Ajouter les valeurs au-dessus des barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Configurer le graphique
        ax.set_xlabel('Modèle', fontsize=12)
        ax.set_ylabel(f'Moyenne {metric.upper()}', fontsize=12)
        ax.set_title(f'Comparaison des modèles sur la métrique {metric.upper()}', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Ajuster la rotation des étiquettes
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Sauvegarder le graphique
        output_path = self.output_dir / f"metric_comparison_{metric}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.debug(f"Graphique sauvegardé dans {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_threshold_comparison(self, metric: str = 'iou', show: bool = False) -> None:
        """
        Génère un graphique comparant les performances des modèles à différents seuils.
        
        Args:
            metric: Métrique à visualiser ('iou', 'dice', etc.).
            show: Si True, affiche le graphique en plus de le sauvegarder.
        """
        logger.debug(f"Génération du graphique de comparaison par seuil pour la métrique '{metric}'")
        
        # Récupérer les données
        model_data = {}
        
        for model_name, model_metrics in self.results.get('models', {}).items():
            threshold_values = []
            metric_values = []
            
            for threshold in self.threshold_values:
                key = f"{metric}_threshold_{threshold}"
                if key in model_metrics.get('test_metrics', {}):
                    threshold_values.append(threshold)
                    metric_values.append(model_metrics['test_metrics'][key])
            
            if threshold_values:
                model_data[model_name] = (threshold_values, metric_values)
        
        if not model_data:
            logger.warning(f"Aucune donnée disponible pour la métrique '{metric}' par seuil")
            return
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (model_name, (thresholds, values)) in enumerate(model_data.items()):
            ax.plot(thresholds, values, marker='o', linewidth=2, 
                    label=model_name, color=self.color_palette[i % len(self.color_palette)])
        
        # Configurer le graphique
        ax.set_xlabel('Seuil de hauteur (m)', fontsize=12)
        ax.set_ylabel(f'{metric.upper()}', fontsize=12)
        ax.set_title(f'Comparaison des modèles sur la métrique {metric.upper()} par seuil', fontsize=14)
        ax.grid(linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        plt.tight_layout()
        
        # Sauvegarder le graphique
        output_path = self.output_dir / f"threshold_comparison_{metric}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.debug(f"Graphique sauvegardé dans {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_training_curves(self, metric: str = 'iou', show: bool = False) -> None:
        """
        Génère des courbes d'apprentissage pour comparer l'évolution des performances.
        
        Args:
            metric: Métrique à visualiser ('iou', 'dice', etc.).
            show: Si True, affiche le graphique en plus de le sauvegarder.
        """
        logger.debug(f"Génération des courbes d'apprentissage pour la métrique '{metric}'")
        
        # Récupérer les données
        model_data = {}
        max_epochs = 0
        
        for model_name, model_metrics in self.results.get('models', {}).items():
            if 'val_history' in model_metrics and metric in model_metrics['val_history']:
                history = model_metrics['val_history'][metric]
                if history:
                    model_data[model_name] = history
                    max_epochs = max(max_epochs, len(history))
        
        if not model_data or max_epochs == 0:
            logger.warning(f"Aucune donnée d'historique disponible pour la métrique '{metric}'")
            return
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, (model_name, history) in enumerate(model_data.items()):
            epochs = list(range(1, len(history) + 1))
            ax.plot(epochs, history, marker='o', linewidth=2, markersize=3,
                    label=model_name, color=self.color_palette[i % len(self.color_palette)])
        
        # Configurer le graphique
        ax.set_xlabel('Époque', fontsize=12)
        ax.set_ylabel(f'{metric.upper()}', fontsize=12)
        ax.set_title(f'Courbes d\'apprentissage - {metric.upper()}', fontsize=14)
        ax.grid(linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        # Définir les limites
        ax.set_xlim(1, max_epochs)
        
        plt.tight_layout()
        
        # Sauvegarder le graphique
        output_path = self.output_dir / f"training_curves_{metric}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.debug(f"Graphique sauvegardé dans {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_training_time_comparison(self, show: bool = False) -> None:
        """
        Génère un graphique comparant les temps d'entraînement des modèles.
        
        Args:
            show: Si True, affiche le graphique en plus de le sauvegarder.
        """
        logger.debug("Génération du graphique de comparaison des temps d'entraînement")
        
        # Récupérer les données
        model_names = []
        training_times = []
        
        for model_name, model_data in self.results.get('models', {}).items():
            if 'training_time' in model_data:
                model_names.append(model_name)
                # Convertir en minutes
                training_times.append(model_data['training_time'] / 60.0)
        
        if not model_names:
            logger.warning("Aucune donnée de temps d'entraînement disponible")
            return
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(model_names, training_times, color=self.color_palette[:len(model_names)])
        
        # Ajouter les valeurs au-dessus des barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f} min', ha='center', va='bottom')
        
        # Configurer le graphique
        ax.set_xlabel('Modèle', fontsize=12)
        ax.set_ylabel('Temps d\'entraînement (minutes)', fontsize=12)
        ax.set_title('Comparaison des temps d\'entraînement des modèles', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Ajuster la rotation des étiquettes
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Sauvegarder le graphique
        output_path = self.output_dir / "training_time_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.debug(f"Graphique sauvegardé dans {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_convergence_speed_comparison(self, metric: str = 'iou', show: bool = False) -> None:
        """
        Génère un graphique comparant la vitesse de convergence des modèles.
        
        Args:
            metric: Métrique à utiliser pour la vitesse de convergence.
            show: Si True, affiche le graphique en plus de le sauvegarder.
        """
        logger.debug(f"Génération du graphique de comparaison des vitesses de convergence pour '{metric}'")
        
        # Récupérer les données
        model_names = []
        convergence_speeds = []
        
        for model_name, model_summary in self.results.get('summary', {}).get('models', {}).items():
            if 'convergence_speed' in model_summary and metric in model_summary['convergence_speed']:
                speed = model_summary['convergence_speed'][metric]
                if speed > 0:  # Ignorer les valeurs négatives (non convergence)
                    model_names.append(model_name)
                    convergence_speeds.append(speed)
        
        if not model_names:
            logger.warning(f"Aucune donnée de vitesse de convergence disponible pour '{metric}'")
            return
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(model_names, convergence_speeds, color=self.color_palette[:len(model_names)])
        
        # Ajouter les valeurs au-dessus des barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)} époques', ha='center', va='bottom')
        
        # Configurer le graphique
        ax.set_xlabel('Modèle', fontsize=12)
        ax.set_ylabel('Époques pour convergence', fontsize=12)
        ax.set_title(f'Vitesse de convergence des modèles ({metric.upper()})', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Ajuster la rotation des étiquettes
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Sauvegarder le graphique
        output_path = self.output_dir / f"convergence_speed_{metric}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.debug(f"Graphique sauvegardé dans {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_radar_chart(self, metrics: List[str] = None, show: bool = False) -> None:
        """
        Génère un graphique radar (toile d'araignée) pour comparer les modèles 
        sur plusieurs métriques simultanément.
        
        Args:
            metrics: Liste des métriques à inclure dans le radar (par défaut, toutes).
            show: Si True, affiche le graphique en plus de le sauvegarder.
        """
        metrics = metrics or self.metrics
        logger.debug(f"Génération du graphique radar avec les métriques: {metrics}")
        
        # Récupérer les données
        model_data = {}
        
        for model_name, model_summary in self.results.get('summary', {}).get('models', {}).items():
            metric_values = []
            
            for metric in metrics:
                key = f"{metric}_average"
                if key in model_summary:
                    metric_values.append(model_summary[key])
                else:
                    metric_values.append(0)
            
            if any(metric_values):  # Au moins une métrique doit être non nulle
                model_data[model_name] = metric_values
        
        if not model_data:
            logger.warning("Aucune donnée disponible pour le graphique radar")
            return
        
        # Nombre de variables
        N = len(metrics)
        
        # Angles pour chaque axe
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        # Fermer le graphique en répétant le premier angle
        angles += angles[:1]
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for i, (model_name, values) in enumerate(model_data.items()):
            # Fermer le graphique en répétant la première valeur
            values_closed = values + values[:1]
            ax.plot(angles, values_closed, 'o-', linewidth=2, 
                    label=model_name, color=self.color_palette[i % len(self.color_palette)])
            ax.fill(angles, values_closed, alpha=0.1, 
                    color=self.color_palette[i % len(self.color_palette)])
        
        # Étiquettes des axes
        metric_labels = [metric.upper() for metric in metrics]
        plt.xticks(angles[:-1], metric_labels)
        
        # Ajouter la légende
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Comparaison des modèles sur plusieurs métriques', fontsize=14)
        
        # Sauvegarder le graphique
        output_path = self.output_dir / "radar_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.debug(f"Graphique sauvegardé dans {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close() 