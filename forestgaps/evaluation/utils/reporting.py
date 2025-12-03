"""
Module de génération de rapports pour l'évaluation.

Fournit des fonctions pour générer des rapports d'évaluation,
sauvegarder les métriques et créer des comparaisons de sites.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime


def generate_evaluation_report(
    metrics: Dict[str, Any],
    output_path: str,
    model_name: str = "model",
    site_name: str = "site",
    additional_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Génère un rapport d'évaluation complet.

    Args:
        metrics: Dictionnaire de métriques d'évaluation
        output_path: Chemin où sauvegarder le rapport
        model_name: Nom du modèle évalué
        site_name: Nom du site évalué
        additional_info: Informations supplémentaires (optionnel)

    Returns:
        Chemin vers le rapport généré
    """
    report = {
        "model_name": model_name,
        "site_name": site_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics
    }

    if additional_info:
        report["additional_info"] = additional_info

    # Sauvegarder en JSON
    json_path = output_path if output_path.endswith('.json') else f"{output_path}.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Créer aussi un rapport markdown
    md_path = json_path.replace('.json', '.md')
    with open(md_path, 'w') as f:
        f.write(f"# Rapport d'Évaluation - {model_name}\n\n")
        f.write(f"**Site:** {site_name}\n")
        f.write(f"**Date:** {report['timestamp']}\n\n")
        f.write("## Métriques\n\n")

        for key, value in metrics.items():
            if isinstance(value, dict):
                f.write(f"### {key}\n\n")
                for subkey, subvalue in value.items():
                    f.write(f"- **{subkey}:** {subvalue:.4f}\n")
                f.write("\n")
            else:
                f.write(f"- **{key}:** {value:.4f}\n")

    return json_path


def save_metrics_to_csv(
    metrics: Dict[str, Dict[str, float]],
    output_path: str,
    index_name: str = "model"
) -> str:
    """
    Sauvegarde les métriques dans un fichier CSV.

    Args:
        metrics: Dictionnaire de métriques {modèle: {métrique: valeur}}
        output_path: Chemin où sauvegarder le CSV
        index_name: Nom de la colonne d'index

    Returns:
        Chemin vers le fichier CSV créé
    """
    df = pd.DataFrame(metrics).T
    df.index.name = index_name

    csv_path = output_path if output_path.endswith('.csv') else f"{output_path}.csv"
    df.to_csv(csv_path)

    return csv_path


def create_site_comparison(
    site_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    threshold: float = 5.0
) -> str:
    """
    Crée une comparaison entre plusieurs sites.

    Args:
        site_results: Résultats pour chaque site {nom_site: résultats}
        output_dir: Répertoire où sauvegarder la comparaison
        threshold: Seuil de hauteur utilisé

    Returns:
        Chemin vers le rapport de comparaison
    """
    os.makedirs(output_dir, exist_ok=True)

    # Créer un DataFrame avec les métriques principales
    comparison_data = {}
    for site_name, results in site_results.items():
        if 'metrics' in results:
            comparison_data[site_name] = results['metrics']

    # Sauvegarder en CSV
    csv_path = os.path.join(output_dir, f'site_comparison_threshold_{threshold}m.csv')
    save_metrics_to_csv(comparison_data, csv_path, index_name='site')

    # Créer un rapport markdown
    md_path = os.path.join(output_dir, f'site_comparison_threshold_{threshold}m.md')
    with open(md_path, 'w') as f:
        f.write(f"# Comparaison des Sites - Seuil {threshold}m\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Résultats par Site\n\n")

        df = pd.DataFrame(comparison_data).T
        f.write(df.to_markdown())
        f.write("\n")

    return md_path


def generate_comparison_report(
    results: Dict[str, Any],
    output_dir: str,
    thresholds: List[float]
) -> str:
    """
    Génère un rapport de comparaison entre plusieurs modèles.

    Args:
        results: Résultats pour chaque modèle {nom_modèle: résultats}
        output_dir: Répertoire où sauvegarder le rapport
        thresholds: Liste des seuils évalués

    Returns:
        Chemin vers le rapport de comparaison
    """
    os.makedirs(output_dir, exist_ok=True)

    # Créer un rapport pour chaque seuil
    for threshold in thresholds:
        threshold_data = {}
        for model_name, result in results.items():
            if hasattr(result, 'metrics') and str(threshold) in result.metrics:
                threshold_data[model_name] = result.metrics[str(threshold)]

        if threshold_data:
            csv_path = os.path.join(output_dir, f'model_comparison_threshold_{threshold}m.csv')
            save_metrics_to_csv(threshold_data, csv_path, index_name='model')

    # Créer un rapport markdown global
    report_path = os.path.join(output_dir, 'model_comparison_report.md')
    with open(report_path, 'w') as f:
        f.write("# Rapport de Comparaison des Modèles\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Seuils évalués:** {', '.join(map(str, thresholds))}m\n\n")

        for model_name, result in results.items():
            f.write(f"## {model_name}\n\n")
            if hasattr(result, 'metrics'):
                for threshold in thresholds:
                    threshold_key = str(threshold)
                    if threshold_key in result.metrics:
                        f.write(f"### Seuil {threshold}m\n\n")
                        for metric, value in result.metrics[threshold_key].items():
                            f.write(f"- **{metric}:** {value:.4f}\n")
                        f.write("\n")

    return report_path


__all__ = [
    "generate_evaluation_report",
    "save_metrics_to_csv",
    "create_site_comparison",
    "generate_comparison_report"
]
