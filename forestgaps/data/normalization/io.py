"""
Module d'entrées/sorties pour les statistiques de normalisation.

Ce module fournit des fonctionnalités pour charger, sauvegarder et 
convertir des statistiques de normalisation entre différents formats.
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import datetime

import numpy as np
import torch
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from forestgaps.data.normalization.statistics import NormalizationStatistics

# Configuration du logger
logger = logging.getLogger(__name__)


def save_stats_json(stats: NormalizationStatistics, output_path: str) -> None:
    """
    Sauvegarde des statistiques de normalisation au format JSON.
    
    Args:
        stats: Statistiques de normalisation à sauvegarder
        output_path: Chemin où sauvegarder le fichier JSON
    """
    stats.save(output_path)


def load_stats_json(input_path: str) -> NormalizationStatistics:
    """
    Charge des statistiques de normalisation depuis un fichier JSON.
    
    Args:
        input_path: Chemin du fichier JSON à charger
        
    Returns:
        Statistiques de normalisation chargées
    """
    return NormalizationStatistics.load(input_path)


def save_stats_pickle(stats: NormalizationStatistics, output_path: str) -> None:
    """
    Sauvegarde des statistiques de normalisation au format pickle.
    
    Args:
        stats: Statistiques de normalisation à sauvegarder
        output_path: Chemin où sauvegarder le fichier pickle
    """
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(stats, f)
        logger.info(f"Statistiques sauvegardées au format pickle dans {output_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des statistiques au format pickle: {str(e)}")


def load_stats_pickle(input_path: str) -> NormalizationStatistics:
    """
    Charge des statistiques de normalisation depuis un fichier pickle.
    
    Args:
        input_path: Chemin du fichier pickle à charger
        
    Returns:
        Statistiques de normalisation chargées
    """
    try:
        with open(input_path, 'rb') as f:
            stats = pickle.load(f)
        logger.info(f"Statistiques chargées depuis {input_path}")
        return stats
    except Exception as e:
        logger.error(f"Erreur lors du chargement des statistiques depuis {input_path}: {str(e)}")
        raise


def stats_to_dataframe(stats: NormalizationStatistics) -> pd.DataFrame:
    """
    Convertit des statistiques de normalisation en DataFrame pandas.
    
    Args:
        stats: Statistiques de normalisation à convertir
        
    Returns:
        DataFrame contenant les statistiques
    """
    # Extrait les statistiques globales
    global_stats = stats.stats['global']
    df_data = {
        'stat_name': [],
        'value': []
    }
    
    # Ajoute les statistiques globales
    for key, value in global_stats.items():
        if key not in ['hist_bins', 'hist_values'] and value is not None:
            df_data['stat_name'].append(key)
            df_data['value'].append(value)
    
    # Ajoute les métadonnées
    df_data['stat_name'].append('method')
    df_data['value'].append(stats.method)
    
    # Crée le DataFrame
    return pd.DataFrame(df_data)


def stats_to_csv(stats: NormalizationStatistics, output_path: str) -> None:
    """
    Sauvegarde des statistiques de normalisation au format CSV.
    
    Args:
        stats: Statistiques de normalisation à sauvegarder
        output_path: Chemin où sauvegarder le fichier CSV
    """
    try:
        df = stats_to_dataframe(stats)
        df.to_csv(output_path, index=False)
        logger.info(f"Statistiques sauvegardées au format CSV dans {output_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des statistiques au format CSV: {str(e)}")


def plot_stats_histogram(
    stats: NormalizationStatistics,
    output_path: Optional[str] = None,
    figure_size: Tuple[int, int] = (10, 6),
    dpi: int = 100,
    log_scale: bool = False,
    overlay_method: bool = True,
    show_percentiles: bool = True
) -> plt.Figure:
    """
    Génère un histogramme à partir des statistiques de normalisation.
    
    Args:
        stats: Statistiques de normalisation
        output_path: Chemin où sauvegarder l'image (optionnel)
        figure_size: Taille de la figure (largeur, hauteur)
        dpi: Résolution de l'image
        log_scale: Utilise une échelle logarithmique pour l'axe y
        overlay_method: Superpose les lignes de la méthode de normalisation
        show_percentiles: Affiche les lignes des percentiles
        
    Returns:
        Figure matplotlib
    """
    global_stats = stats.stats['global']
    
    # Vérifie si l'histogramme est disponible
    if 'hist_bins' not in global_stats or 'hist_values' not in global_stats:
        logger.warning("Aucun histogramme disponible dans les statistiques")
        return None
    
    # Récupère les bins et valeurs de l'histogramme
    hist_bins = global_stats['hist_bins']
    hist_values = global_stats['hist_values']
    
    # Crée la figure
    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    
    # Trace l'histogramme
    ax.bar(
        hist_bins[:-1],
        hist_values,
        width=hist_bins[1] - hist_bins[0],
        alpha=0.7,
        color='skyblue',
        edgecolor='steelblue',
        label='Distribution'
    )
    
    # Applique l'échelle logarithmique si demandé
    if log_scale:
        ax.set_yscale('log')
    
    # Superpose les lignes de la méthode de normalisation
    if overlay_method:
        method = stats.method
        if method == 'minmax':
            ax.axvline(x=global_stats['min'], color='red', linestyle='--', label='Min')
            ax.axvline(x=global_stats['max'], color='darkred', linestyle='--', label='Max')
        elif method == 'zscore':
            ax.axvline(x=global_stats['mean'], color='green', linestyle='--', label='Moyenne')
            ax.axvline(x=global_stats['mean'] - global_stats['std'], color='lightgreen', linestyle=':', label='Moyenne - Écart-type')
            ax.axvline(x=global_stats['mean'] + global_stats['std'], color='darkgreen', linestyle=':', label='Moyenne + Écart-type')
        elif method == 'robust' or method == 'percentile':
            ax.axvline(x=global_stats['median'], color='purple', linestyle='--', label='Médiane')
    
    # Affiche les percentiles
    if show_percentiles:
        ax.axvline(x=global_stats['p1'], color='orange', linestyle=':', label='1%')
        ax.axvline(x=global_stats['p99'], color='darkorange', linestyle=':', label='99%')
    
    # Configure les axes et légendes
    ax.set_xlabel('Valeur')
    ax.set_ylabel('Fréquence')
    ax.set_title(f'Distribution des valeurs (méthode: {stats.method})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ajoute des informations sur les statistiques
    stats_text = f"Min: {global_stats['min']:.2f}, Max: {global_stats['max']:.2f}\n"
    stats_text += f"Moyenne: {global_stats['mean']:.2f}, Écart-type: {global_stats['std']:.2f}\n"
    stats_text += f"Médiane: {global_stats['median']:.2f}\n"
    stats_text += f"P1: {global_stats['p1']:.2f}, P99: {global_stats['p99']:.2f}"
    
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Ajuste la mise en page
    plt.tight_layout()
    
    # Sauvegarde l'image si un chemin est spécifié
    if output_path:
        try:
            plt.savefig(output_path, dpi=dpi)
            logger.info(f"Histogramme sauvegardé dans {output_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'histogramme: {str(e)}")
    
    return fig


def generate_stats_report(
    stats: NormalizationStatistics,
    output_dir: str,
    prefix: str = "norm_stats",
    include_histogram: bool = True,
    include_csv: bool = True,
    include_json: bool = True
) -> Dict[str, str]:
    """
    Génère un rapport complet des statistiques de normalisation.
    
    Args:
        stats: Statistiques de normalisation
        output_dir: Répertoire où sauvegarder les fichiers
        prefix: Préfixe pour les noms de fichiers
        include_histogram: Inclut un histogramme dans le rapport
        include_csv: Inclut un fichier CSV dans le rapport
        include_json: Inclut un fichier JSON dans le rapport
        
    Returns:
        Dictionnaire des chemins des fichiers générés
    """
    # Crée le répertoire si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Génère un timestamp pour les noms de fichiers
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{prefix}_{timestamp}"
    
    result_paths = {}
    
    # Génère le fichier JSON
    if include_json:
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        save_stats_json(stats, json_path)
        result_paths['json'] = json_path
    
    # Génère le fichier CSV
    if include_csv:
        csv_path = os.path.join(output_dir, f"{base_filename}.csv")
        stats_to_csv(stats, csv_path)
        result_paths['csv'] = csv_path
    
    # Génère l'histogramme
    if include_histogram:
        hist_path = os.path.join(output_dir, f"{base_filename}_hist.png")
        plot_stats_histogram(stats, hist_path)
        result_paths['histogram'] = hist_path
    
    # Génère un rapport textuel
    report_path = os.path.join(output_dir, f"{base_filename}_report.txt")
    try:
        with open(report_path, 'w') as f:
            global_stats = stats.stats['global']
            
            f.write(f"Rapport de statistiques de normalisation - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Informations générales:\n")
            f.write(f"- Méthode de normalisation: {stats.method}\n")
            if stats.method == 'percentile':
                f.write(f"- Plage de percentiles: {stats.percentile_range}\n")
            f.write(f"- Nombre d'échantillons: {global_stats['sample_count']}\n")
            f.write(f"- Nombre de pixels: {global_stats['pixel_count']}\n\n")
            
            f.write("Statistiques globales:\n")
            f.write(f"- Minimum: {global_stats['min']}\n")
            f.write(f"- Maximum: {global_stats['max']}\n")
            f.write(f"- Moyenne: {global_stats['mean']}\n")
            f.write(f"- Écart-type: {global_stats['std']}\n")
            f.write(f"- Médiane: {global_stats['median']}\n")
            f.write(f"- 1er percentile: {global_stats['p1']}\n")
            f.write(f"- 99ème percentile: {global_stats['p99']}\n\n")
            
            f.write("Fichiers générés:\n")
            for key, path in result_paths.items():
                f.write(f"- {key}: {os.path.basename(path)}\n")
        
        result_paths['report'] = report_path
        logger.info(f"Rapport textuel sauvegardé dans {report_path}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération du rapport textuel: {str(e)}")
    
    return result_paths


def compare_stats(
    stats_list: List[NormalizationStatistics],
    labels: List[str],
    output_path: Optional[str] = None,
    figure_size: Tuple[int, int] = (12, 8),
    dpi: int = 100
) -> plt.Figure:
    """
    Compare plusieurs ensembles de statistiques de normalisation.
    
    Args:
        stats_list: Liste des statistiques de normalisation à comparer
        labels: Étiquettes pour chaque ensemble de statistiques
        output_path: Chemin où sauvegarder l'image (optionnel)
        figure_size: Taille de la figure (largeur, hauteur)
        dpi: Résolution de l'image
        
    Returns:
        Figure matplotlib
    """
    if len(stats_list) != len(labels):
        raise ValueError("Le nombre d'ensembles de statistiques doit être égal au nombre d'étiquettes")
    
    # Crée la figure avec deux sous-graphes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size, dpi=dpi)
    
    # Prépare les données pour les boxplots
    data_ranges = []
    means = []
    medians = []
    stds = []
    p1s = []
    p99s = []
    
    for stats in stats_list:
        global_stats = stats.stats['global']
        data_ranges.append([global_stats['min'], global_stats['max']])
        means.append(global_stats['mean'])
        medians.append(global_stats['median'])
        stds.append(global_stats['std'])
        p1s.append(global_stats['p1'])
        p99s.append(global_stats['p99'])
    
    # Boxplot simple
    ax1.boxplot(
        data_ranges,
        labels=labels,
        showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='blue', markeredgecolor='blue')
    )
    ax1.set_title('Comparaison des plages de valeurs')
    ax1.set_ylabel('Valeur')
    ax1.grid(True, alpha=0.3)
    
    # Graphique en barres des statistiques clés
    x = np.arange(len(labels))
    width = 0.15
    
    ax2.bar(x - 2*width, means, width, label='Moyenne', color='skyblue')
    ax2.bar(x - width, medians, width, label='Médiane', color='lightgreen')
    ax2.bar(x, stds, width, label='Écart-type', color='salmon')
    ax2.bar(x + width, p1s, width, label='P1', color='lightyellow')
    ax2.bar(x + 2*width, p99s, width, label='P99', color='lightpink')
    
    ax2.set_xlabel('Ensemble de données')
    ax2.set_ylabel('Valeur')
    ax2.set_title('Comparaison des statistiques clés')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Ajuste la mise en page
    plt.tight_layout()
    
    # Sauvegarde l'image si un chemin est spécifié
    if output_path:
        try:
            plt.savefig(output_path, dpi=dpi)
            logger.info(f"Comparaison sauvegardée dans {output_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la comparaison: {str(e)}")
    
    return fig


def merge_stats(
    stats_list: List[NormalizationStatistics],
    weights: Optional[List[float]] = None,
    method: Optional[str] = None
) -> NormalizationStatistics:
    """
    Fusionne plusieurs ensembles de statistiques de normalisation.
    
    Args:
        stats_list: Liste des statistiques de normalisation à fusionner
        weights: Poids à appliquer à chaque ensemble (None = poids égaux)
        method: Méthode de normalisation pour le résultat (None = méthode du premier ensemble)
        
    Returns:
        Statistiques de normalisation fusionnées
    """
    if not stats_list:
        raise ValueError("La liste des statistiques à fusionner ne doit pas être vide")
    
    # Utilise des poids égaux si non spécifiés
    if weights is None:
        weights = [1.0 / len(stats_list)] * len(stats_list)
    
    # Normalise les poids
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Utilise la méthode du premier ensemble si non spécifiée
    method = method or stats_list[0].method
    
    # Crée un nouvel ensemble de statistiques
    merged_stats = NormalizationStatistics(method=method)
    
    # Fusionne les statistiques globales
    global_merged = {
        'min': 0.0,
        'max': 0.0,
        'mean': 0.0,
        'std': 0.0,
        'median': 0.0,
        'p1': 0.0,
        'p99': 0.0,
        'sample_count': 0,
        'pixel_count': 0
    }
    
    # Calcule la moyenne pondérée des statistiques
    for i, stats in enumerate(stats_list):
        weight = weights[i]
        global_stats = stats.stats['global']
        
        for key in ['min', 'max', 'mean', 'median', 'p1', 'p99']:
            global_merged[key] += global_stats[key] * weight
        
        # Pour l'écart-type, on utilise la variance pondérée
        global_merged['std'] += (global_stats['std'] ** 2) * weight
        
        global_merged['sample_count'] += global_stats['sample_count']
        global_merged['pixel_count'] += global_stats['pixel_count']
    
    # Finalise l'écart-type
    global_merged['std'] = np.sqrt(global_merged['std'])
    
    # Met à jour les statistiques fusionnées
    merged_stats.stats['global'] = global_merged
    merged_stats.stats['method'] = method
    
    return merged_stats


def export_stats_to_onnx(
    stats: NormalizationStatistics,
    output_path: str,
    method: Optional[str] = None
) -> None:
    """
    Exporte les statistiques de normalisation pour utilisation avec ONNX.
    
    Args:
        stats: Statistiques de normalisation
        output_path: Chemin où sauvegarder le fichier ONNX
        method: Méthode de normalisation (None = utilise la méthode des statistiques)
    """
    import onnx
    from onnx import helper, TensorProto
    
    method = method or stats.method
    global_stats = stats.stats['global']
    
    try:
        # Crée les nœuds ONNX selon la méthode de normalisation
        if method == 'minmax':
            # Normalisation min-max: (x - min) / (max - min)
            min_val = np.array([global_stats['min']], dtype=np.float32)
            max_val = np.array([global_stats['max']], dtype=np.float32)
            range_val = np.array([max_val[0] - min_val[0]], dtype=np.float32)
            
            # Crée les constantes
            min_node = helper.make_node(
                'Constant',
                inputs=[],
                outputs=['min_val'],
                value=helper.make_tensor(
                    name='min_val_tensor',
                    data_type=TensorProto.FLOAT,
                    dims=[1],
                    vals=min_val
                )
            )
            
            range_node = helper.make_node(
                'Constant',
                inputs=[],
                outputs=['range_val'],
                value=helper.make_tensor(
                    name='range_val_tensor',
                    data_type=TensorProto.FLOAT,
                    dims=[1],
                    vals=range_val
                )
            )
            
            # Crée les opérations
            sub_node = helper.make_node(
                'Sub',
                inputs=['input', 'min_val'],
                outputs=['sub_output']
            )
            
            div_node = helper.make_node(
                'Div',
                inputs=['sub_output', 'range_val'],
                outputs=['normalized_output']
            )
            
            # Crée le graphe
            graph_def = helper.make_graph(
                [min_node, range_node, sub_node, div_node],
                'minmax_normalization',
                [helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, None, None])],
                [helper.make_tensor_value_info('normalized_output', TensorProto.FLOAT, [None, None, None])]
            )
            
        elif method == 'zscore':
            # Normalisation z-score: (x - mean) / std
            mean_val = np.array([global_stats['mean']], dtype=np.float32)
            std_val = np.array([global_stats['std']], dtype=np.float32)
            
            # Crée les constantes
            mean_node = helper.make_node(
                'Constant',
                inputs=[],
                outputs=['mean_val'],
                value=helper.make_tensor(
                    name='mean_val_tensor',
                    data_type=TensorProto.FLOAT,
                    dims=[1],
                    vals=mean_val
                )
            )
            
            std_node = helper.make_node(
                'Constant',
                inputs=[],
                outputs=['std_val'],
                value=helper.make_tensor(
                    name='std_val_tensor',
                    data_type=TensorProto.FLOAT,
                    dims=[1],
                    vals=std_val
                )
            )
            
            # Crée les opérations
            sub_node = helper.make_node(
                'Sub',
                inputs=['input', 'mean_val'],
                outputs=['sub_output']
            )
            
            div_node = helper.make_node(
                'Div',
                inputs=['sub_output', 'std_val'],
                outputs=['normalized_output']
            )
            
            # Crée le graphe
            graph_def = helper.make_graph(
                [mean_node, std_node, sub_node, div_node],
                'zscore_normalization',
                [helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, None, None])],
                [helper.make_tensor_value_info('normalized_output', TensorProto.FLOAT, [None, None, None])]
            )
            
        else:
            raise ValueError(f"Méthode de normalisation '{method}' non supportée pour l'export ONNX")
        
        # Crée le modèle ONNX
        model_def = helper.make_model(
            graph_def,
            producer_name='forestgaps_normalization'
        )
        
        # Vérifie et sauvegarde le modèle
        onnx.checker.check_model(model_def)
        onnx.save(model_def, output_path)
        
        logger.info(f"Modèle de normalisation ONNX sauvegardé dans {output_path}")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'export des statistiques au format ONNX: {str(e)}")
        raise 