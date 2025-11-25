"""
Module de génération de rapports pour les résultats de benchmarking.

Ce module fournit des outils pour générer des rapports détaillés à partir
des résultats de benchmarking des modèles.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from tabulate import tabulate

logger = logging.getLogger(__name__)


class BenchmarkReporter:
    """
    Classe pour générer des rapports détaillés sur les résultats de benchmarking.
    
    Cette classe crée des rapports textuels et HTML à partir des résultats
    de benchmarking, facilitant l'analyse et la comparaison des modèles.
    """
    
    def __init__(self, results: Dict[str, Any], output_dir: Union[str, Path]):
        """
        Initialise le générateur de rapports de benchmarking.
        
        Args:
            results: Résultats de benchmarking (sortie de ModelComparison.run()).
            output_dir: Répertoire où sauvegarder les rapports.
        """
        self.results = results
        self.output_dir = Path(output_dir) / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extraire les configurations
        self.metrics = results.get('config', {}).get('metrics', ['iou', 'dice', 'accuracy'])
        self.threshold_values = results.get('config', {}).get('threshold_values', [2.0, 5.0, 10.0, 15.0])
        
        logger.info(f"Initialisé BenchmarkReporter avec répertoire de sortie: {self.output_dir}")
    
    def generate_report(self, format: str = 'all') -> None:
        """
        Génère un rapport détaillé des résultats de benchmarking.
        
        Args:
            format: Format du rapport ('text', 'html', 'markdown', 'all').
        """
        logger.info(f"Génération du rapport de benchmarking au format {format}")
        
        # Générer le contenu du rapport
        report_content = self._generate_report_content()
        
        # Sauvegarder dans les formats demandés
        if format in ['text', 'all']:
            self._save_text_report(report_content)
        
        if format in ['html', 'all']:
            self._save_html_report(report_content)
        
        if format in ['markdown', 'all']:
            self._save_markdown_report(report_content)
        
        logger.info("Génération du rapport terminée")
    
    def _generate_report_content(self) -> Dict[str, Any]:
        """
        Génère le contenu du rapport de benchmarking.
        
        Returns:
            Dictionnaire contenant les différentes sections du rapport.
        """
        content = {}
        
        # Informations générales
        content['title'] = "Rapport de Benchmarking des Modèles ForestGaps"
        content['date'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        content['description'] = "Ce rapport présente les résultats de la comparaison de différents modèles de détection de trouées forestières."
        
        # Résumé des résultats
        content['summary'] = self._generate_summary()
        
        # Tableaux des métriques
        content['metrics_tables'] = self._generate_metrics_tables()
        
        # Résultats par modèle
        content['model_results'] = self._generate_model_results()
        
        # Résultats par seuil
        content['threshold_results'] = self._generate_threshold_results()
        
        # Meilleurs modèles
        content['best_models'] = self._generate_best_models_section()
        
        # Analyse du temps d'entraînement
        content['training_time'] = self._generate_training_time_section()
        
        # Analyse de convergence
        content['convergence'] = self._generate_convergence_section()
        
        # Paramètres des modèles
        content['model_params'] = self._generate_model_params_section()
        
        # Configuration utilisée
        content['config'] = self._generate_config_section()
        
        return content
    
    def _generate_summary(self) -> str:
        """
        Génère un résumé global des résultats.
        
        Returns:
            Chaîne de caractères contenant le résumé.
        """
        summary = []
        
        # Nombre de modèles
        model_count = len(self.results.get('models', {}))
        summary.append(f"Nombre de modèles comparés: {model_count}")
        
        # Métriques analysées
        metrics_str = ", ".join(self.metrics)
        summary.append(f"Métriques analysées: {metrics_str}")
        
        # Seuils de hauteur
        thresholds_str = ", ".join([f"{t}m" for t in self.threshold_values])
        summary.append(f"Seuils de hauteur analysés: {thresholds_str}")
        
        # Meilleur modèle global
        best_models = self.results.get('best_models', {})
        
        for metric in self.metrics:
            best_key = f"{metric}_average"
            if best_key in best_models:
                summary.append(f"Meilleur modèle pour {metric.upper()}: {best_models[best_key]}")
        
        if 'training_time' in best_models:
            summary.append(f"Modèle le plus rapide à entraîner: {best_models['training_time']}")
        
        return "\n".join(summary)
    
    def _generate_metrics_tables(self) -> Dict[str, str]:
        """
        Génère des tableaux de métriques pour tous les modèles.
        
        Returns:
            Dictionnaire contenant les tableaux pour chaque métrique.
        """
        tables = {}
        
        # Pour chaque métrique
        for metric in self.metrics:
            # Créer un DataFrame pour cette métrique
            data = []
            columns = ['Modèle', 'Moyenne'] + [f"Seuil {t}m" for t in self.threshold_values]
            
            for model_name, model_data in self.results.get('models', {}).items():
                row = [model_name]
                
                # Moyenne pour cette métrique
                summary = self.results.get('summary', {}).get('models', {}).get(model_name, {})
                avg_key = f"{metric}_average"
                
                if avg_key in summary:
                    row.append(f"{summary[avg_key]:.4f}")
                else:
                    row.append("N/A")
                
                # Valeurs par seuil
                for threshold in self.threshold_values:
                    key = f"{metric}_threshold_{threshold}"
                    
                    if key in model_data.get('test_metrics', {}):
                        row.append(f"{model_data['test_metrics'][key]:.4f}")
                    else:
                        row.append("N/A")
                
                data.append(row)
            
            # Créer le tableau
            df = pd.DataFrame(data, columns=columns)
            tables[metric] = df
        
        # Convertir les DataFrames en chaînes de caractères
        table_strings = {}
        for metric, df in tables.items():
            table_strings[metric] = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
        
        return table_strings
    
    def _generate_model_results(self) -> Dict[str, str]:
        """
        Génère des résumés détaillés pour chaque modèle.
        
        Returns:
            Dictionnaire contenant les résumés par modèle.
        """
        model_results = {}
        
        for model_name, model_data in self.results.get('models', {}).items():
            summary = []
            
            # Informations générales
            summary.append(f"### Modèle: {model_name}")
            
            # Métriques moyennes
            model_summary = self.results.get('summary', {}).get('models', {}).get(model_name, {})
            summary.append("\n#### Métriques Moyennes:")
            
            for metric in self.metrics:
                avg_key = f"{metric}_average"
                if avg_key in model_summary:
                    summary.append(f"- {metric.upper()}: {model_summary[avg_key]:.4f}")
            
            # Temps d'entraînement
            if 'training_time' in model_data:
                minutes = model_data['training_time'] / 60.0
                summary.append(f"\n#### Temps d'entraînement: {minutes:.2f} minutes")
            
            # Vitesse de convergence
            if 'convergence_speed' in model_summary:
                summary.append("\n#### Vitesse de Convergence:")
                
                for metric, speed in model_summary['convergence_speed'].items():
                    if speed > 0:
                        summary.append(f"- {metric.upper()}: {speed} époques")
            
            # Paramètres du modèle
            if 'model_params' in model_data:
                summary.append("\n#### Paramètres du Modèle:")
                
                for param, value in model_data['model_params'].items():
                    summary.append(f"- {param}: {value}")
            
            model_results[model_name] = "\n".join(summary)
        
        return model_results
    
    def _generate_threshold_results(self) -> Dict[str, str]:
        """
        Génère des résumés des résultats par seuil de hauteur.
        
        Returns:
            Dictionnaire contenant les résumés par seuil.
        """
        threshold_results = {}
        
        for threshold in self.threshold_values:
            summary = []
            
            summary.append(f"### Résultats pour le seuil {threshold}m")
            
            # Tableau des métriques pour ce seuil
            data = []
            columns = ['Modèle'] + [m.upper() for m in self.metrics]
            
            for model_name, model_data in self.results.get('models', {}).items():
                row = [model_name]
                
                for metric in self.metrics:
                    key = f"{metric}_threshold_{threshold}"
                    
                    if key in model_data.get('test_metrics', {}):
                        row.append(f"{model_data['test_metrics'][key]:.4f}")
                    else:
                        row.append("N/A")
                
                data.append(row)
            
            # Créer le tableau
            df = pd.DataFrame(data, columns=columns)
            table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
            
            summary.append("\n" + table)
            
            # Meilleur modèle pour chaque métrique
            summary.append("\n#### Meilleurs modèles pour ce seuil:")
            
            for metric in self.metrics:
                key = f"{metric}_threshold_{threshold}"
                
                if key in self.results.get('best_models', {}):
                    best_model = self.results['best_models'][key]
                    summary.append(f"- {metric.upper()}: {best_model}")
            
            threshold_results[str(threshold)] = "\n".join(summary)
        
        return threshold_results
    
    def _generate_best_models_section(self) -> str:
        """
        Génère une section sur les meilleurs modèles pour chaque métrique.
        
        Returns:
            Chaîne de caractères contenant la section.
        """
        best_models = self.results.get('best_models', {})
        
        if not best_models:
            return "Aucune information sur les meilleurs modèles n'est disponible."
        
        summary = ["### Meilleurs Modèles"]
        
        # Par métrique (moyenne)
        summary.append("\n#### Par Métrique (Moyenne sur tous les seuils):")
        
        for metric in self.metrics:
            key = f"{metric}_average"
            
            if key in best_models:
                summary.append(f"- {metric.upper()}: {best_models[key]}")
        
        # Par métrique et par seuil
        summary.append("\n#### Par Métrique et Seuil:")
        
        for metric in self.metrics:
            metric_summary = []
            
            for threshold in self.threshold_values:
                key = f"{metric}_threshold_{threshold}"
                
                if key in best_models:
                    metric_summary.append(f"  - Seuil {threshold}m: {best_models[key]}")
            
            if metric_summary:
                summary.append(f"- {metric.upper()}:")
                summary.extend(metric_summary)
        
        # Autres critères
        summary.append("\n#### Autres Critères:")
        
        if 'training_time' in best_models:
            summary.append(f"- Temps d'entraînement: {best_models['training_time']}")
        
        for metric in self.metrics:
            key = f"convergence_{metric}"
            
            if key in best_models:
                summary.append(f"- Vitesse de convergence ({metric.upper()}): {best_models[key]}")
        
        return "\n".join(summary)
    
    def _generate_training_time_section(self) -> str:
        """
        Génère une section sur les temps d'entraînement.
        
        Returns:
            Chaîne de caractères contenant la section.
        """
        summary = ["### Analyse des Temps d'Entraînement"]
        
        # Tableau des temps d'entraînement
        data = []
        columns = ['Modèle', 'Temps (minutes)']
        
        for model_name, model_data in self.results.get('models', {}).items():
            if 'training_time' in model_data:
                minutes = model_data['training_time'] / 60.0
                data.append([model_name, f"{minutes:.2f}"])
        
        if not data:
            return "Aucune information sur les temps d'entraînement n'est disponible."
        
        # Créer le tableau
        df = pd.DataFrame(data, columns=columns)
        df = df.sort_values(by='Temps (minutes)')
        
        table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
        summary.append("\n" + table)
        
        return "\n".join(summary)
    
    def _generate_convergence_section(self) -> str:
        """
        Génère une section sur la vitesse de convergence.
        
        Returns:
            Chaîne de caractères contenant la section.
        """
        summary = ["### Analyse de la Vitesse de Convergence"]
        
        # Pour chaque métrique
        for metric in self.metrics:
            metric_summary = [f"\n#### Convergence pour {metric.upper()}:"]
            
            data = []
            columns = ['Modèle', 'Époques pour 90% de performance']
            
            for model_name, model_summary in self.results.get('summary', {}).get('models', {}).items():
                if 'convergence_speed' in model_summary and metric in model_summary['convergence_speed']:
                    speed = model_summary['convergence_speed'][metric]
                    
                    if speed > 0:  # Ignorer les valeurs négatives (non convergence)
                        data.append([model_name, speed])
            
            if data:
                # Créer le tableau
                df = pd.DataFrame(data, columns=columns)
                df = df.sort_values(by='Époques pour 90% de performance')
                
                table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
                metric_summary.append(table)
                
                summary.extend(metric_summary)
        
        if len(summary) == 1:
            return "Aucune information sur la vitesse de convergence n'est disponible."
        
        return "\n".join(summary)
    
    def _generate_model_params_section(self) -> str:
        """
        Génère une section sur les paramètres des modèles.
        
        Returns:
            Chaîne de caractères contenant la section.
        """
        summary = ["### Paramètres des Modèles"]
        
        for model_name, model_data in self.results.get('models', {}).items():
            if 'model_params' in model_data and model_data['model_params']:
                model_summary = [f"\n#### {model_name}:"]
                
                for param, value in model_data['model_params'].items():
                    model_summary.append(f"- {param}: {value}")
                
                summary.extend(model_summary)
        
        if len(summary) == 1:
            return "Aucune information sur les paramètres des modèles n'est disponible."
        
        return "\n".join(summary)
    
    def _generate_config_section(self) -> str:
        """
        Génère une section sur la configuration utilisée.
        
        Returns:
            Chaîne de caractères contenant la section.
        """
        summary = ["### Configuration Utilisée"]
        
        base_config = self.results.get('config', {}).get('base_config', {})
        
        if base_config:
            # Configuration d'entraînement
            if 'training' in base_config:
                summary.append("\n#### Configuration d'Entraînement:")
                
                for key, value in base_config['training'].items():
                    summary.append(f"- {key}: {value}")
            
            # Configuration des données
            if 'data' in base_config:
                summary.append("\n#### Configuration des Données:")
                
                for key, value in base_config['data'].items():
                    summary.append(f"- {key}: {value}")
            
            # Configuration des modèles
            if 'model' in base_config:
                summary.append("\n#### Configuration des Modèles:")
                
                for key, value in base_config['model'].items():
                    summary.append(f"- {key}: {value}")
        else:
            summary.append("\nAucune information de configuration détaillée n'est disponible.")
        
        return "\n".join(summary)
    
    def _save_text_report(self, content: Dict[str, Any]) -> None:
        """
        Sauvegarde le rapport au format texte.
        
        Args:
            content: Contenu du rapport.
        """
        output_path = self.output_dir / "benchmark_report.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Titre et informations générales
            f.write(f"{content['title']}\n")
            f.write("="*len(content['title']) + "\n\n")
            f.write(f"Date: {content['date']}\n\n")
            f.write(f"{content['description']}\n\n")
            
            # Résumé
            f.write("RÉSUMÉ\n")
            f.write("------\n\n")
            f.write(content['summary'] + "\n\n")
            
            # Tableaux des métriques
            f.write("TABLEAUX DES MÉTRIQUES\n")
            f.write("---------------------\n\n")
            
            for metric, table in content['metrics_tables'].items():
                f.write(f"Métrique: {metric.upper()}\n\n")
                f.write(table + "\n\n")
            
            # Résultats par modèle
            f.write("RÉSULTATS PAR MODÈLE\n")
            f.write("-------------------\n\n")
            
            for model_name, results in content['model_results'].items():
                f.write(results.replace('###', '').replace('####', '') + "\n\n")
            
            # Résultats par seuil
            f.write("RÉSULTATS PAR SEUIL\n")
            f.write("-----------------\n\n")
            
            for threshold, results in content['threshold_results'].items():
                f.write(results.replace('###', '').replace('####', '') + "\n\n")
            
            # Meilleurs modèles
            f.write("MEILLEURS MODÈLES\n")
            f.write("----------------\n\n")
            f.write(content['best_models'].replace('###', '').replace('####', '') + "\n\n")
            
            # Temps d'entraînement
            f.write("TEMPS D'ENTRAÎNEMENT\n")
            f.write("-------------------\n\n")
            f.write(content['training_time'].replace('###', '').replace('####', '') + "\n\n")
            
            # Convergence
            f.write("VITESSE DE CONVERGENCE\n")
            f.write("---------------------\n\n")
            f.write(content['convergence'].replace('###', '').replace('####', '') + "\n\n")
            
            # Paramètres des modèles
            f.write("PARAMÈTRES DES MODÈLES\n")
            f.write("---------------------\n\n")
            f.write(content['model_params'].replace('###', '').replace('####', '') + "\n\n")
            
            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-------------\n\n")
            f.write(content['config'].replace('###', '').replace('####', '') + "\n\n")
        
        logger.info(f"Rapport texte sauvegardé dans {output_path}")
    
    def _save_html_report(self, content: Dict[str, Any]) -> None:
        """
        Sauvegarde le rapport au format HTML.
        
        Args:
            content: Contenu du rapport.
        """
        output_path = self.output_dir / "benchmark_report.html"
        
        # Convertir les tableaux Markdown en HTML
        import markdown
        
        html_metrics_tables = {}
        for metric, table in content['metrics_tables'].items():
            html_metrics_tables[metric] = markdown.markdown(table)
        
        html_model_results = {}
        for model_name, results in content['model_results'].items():
            html_model_results[model_name] = markdown.markdown(results)
        
        html_threshold_results = {}
        for threshold, results in content['threshold_results'].items():
            html_threshold_results[threshold] = markdown.markdown(results)
        
        html_best_models = markdown.markdown(content['best_models'])
        html_training_time = markdown.markdown(content['training_time'])
        html_convergence = markdown.markdown(content['convergence'])
        html_model_params = markdown.markdown(content['model_params'])
        html_config = markdown.markdown(content['config'])
        
        # Créer le HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{content['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
        h1, h2, h3, h4 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .section {{ margin-bottom: 30px; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; }}
        .date {{ font-style: italic; margin-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{content['title']}</h1>
            <div class="date">Date: {content['date']}</div>
            <p>{content['description']}</p>
        </div>
        
        <div class="section">
            <h2>Résumé</h2>
            <pre>{content['summary']}</pre>
        </div>
        
        <div class="section">
            <h2>Tableaux des Métriques</h2>
"""
        
        for metric, table_html in html_metrics_tables.items():
            html += f"""
            <h3>Métrique: {metric.upper()}</h3>
            {table_html}
"""
        
        html += """
        </div>
        
        <div class="section">
            <h2>Résultats par Modèle</h2>
"""
        
        for model_name, results_html in html_model_results.items():
            html += f"""
            {results_html}
"""
        
        html += """
        </div>
        
        <div class="section">
            <h2>Résultats par Seuil</h2>
"""
        
        for threshold, results_html in html_threshold_results.items():
            html += f"""
            {results_html}
"""
        
        html += f"""
        </div>
        
        <div class="section">
            <h2>Meilleurs Modèles</h2>
            {html_best_models}
        </div>
        
        <div class="section">
            <h2>Temps d'Entraînement</h2>
            {html_training_time}
        </div>
        
        <div class="section">
            <h2>Vitesse de Convergence</h2>
            {html_convergence}
        </div>
        
        <div class="section">
            <h2>Paramètres des Modèles</h2>
            {html_model_params}
        </div>
        
        <div class="section">
            <h2>Configuration</h2>
            {html_config}
        </div>
    </div>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"Rapport HTML sauvegardé dans {output_path}")
    
    def _save_markdown_report(self, content: Dict[str, Any]) -> None:
        """
        Sauvegarde le rapport au format Markdown.
        
        Args:
            content: Contenu du rapport.
        """
        output_path = self.output_dir / "benchmark_report.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Titre et informations générales
            f.write(f"# {content['title']}\n\n")
            f.write(f"**Date:** {content['date']}\n\n")
            f.write(f"{content['description']}\n\n")
            
            # Résumé
            f.write("## Résumé\n\n")
            f.write(content['summary'] + "\n\n")
            
            # Tableaux des métriques
            f.write("## Tableaux des Métriques\n\n")
            
            for metric, table in content['metrics_tables'].items():
                f.write(f"### Métrique: {metric.upper()}\n\n")
                f.write(table + "\n\n")
            
            # Résultats par modèle
            f.write("## Résultats par Modèle\n\n")
            
            for model_name, results in content['model_results'].items():
                f.write(results + "\n\n")
            
            # Résultats par seuil
            f.write("## Résultats par Seuil\n\n")
            
            for threshold, results in content['threshold_results'].items():
                f.write(results + "\n\n")
            
            # Meilleurs modèles
            f.write("## Meilleurs Modèles\n\n")
            f.write(content['best_models'] + "\n\n")
            
            # Temps d'entraînement
            f.write("## Temps d'Entraînement\n\n")
            f.write(content['training_time'] + "\n\n")
            
            # Convergence
            f.write("## Vitesse de Convergence\n\n")
            f.write(content['convergence'] + "\n\n")
            
            # Paramètres des modèles
            f.write("## Paramètres des Modèles\n\n")
            f.write(content['model_params'] + "\n\n")
            
            # Configuration
            f.write("## Configuration\n\n")
            f.write(content['config'] + "\n\n")
        
        logger.info(f"Rapport Markdown sauvegardé dans {output_path}") 