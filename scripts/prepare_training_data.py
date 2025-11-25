#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de préparation des données pour l'entraînement ForestGaps.

Ce script automatise la préparation des données DSM/CHM pour l'entraînement:
1. Recherche des paires DSM/CHM dans un répertoire
2. Séparation automatique train/test (3 paires train + 1 test par défaut)
3. Vérification et alignement des rasters
4. Génération des masques de trouées
5. Création des tuiles pour l'entraînement
6. Génération d'un fichier de configuration YAML

Usage:
    python prepare_training_data.py --data-dir /path/to/dsm_chm --output-dir /path/to/output

Organisation requise du répertoire data-dir:
    data-dir/
    ├── site1_DSM.tif
    ├── site1_CHM.tif
    ├── site2_DSM.tif
    ├── site2_CHM.tif
    ├── site3_DSM.tif
    ├── site3_CHM.tif
    └── site4_DSM.tif (etc.)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
from collections import defaultdict

# Ajouter le package au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing.alignment import check_alignment, align_rasters
from data.generation.masks import create_binary_mask
from data.generation.tiling import create_tile_grid, extract_tile


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_dsm_chm_pairs(data_dir: Path) -> List[Dict[str, Path]]:
    """
    Trouve automatiquement les paires DSM/CHM dans un répertoire.

    Recherche les fichiers .tif/.tiff contenant 'DSM' et 'CHM' dans leur nom
    et les apparie par préfixe commun.

    Args:
        data_dir: Répertoire contenant les fichiers DSM et CHM

    Returns:
        Liste de dictionnaires avec 'site_name', 'dsm_path', 'chm_path'
    """
    logger.info(f"Recherche des paires DSM/CHM dans {data_dir}")

    # Trouver tous les fichiers TIF
    tif_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff")) + \
                list(data_dir.glob("*.TIF")) + list(data_dir.glob("*.TIFF"))

    # Grouper par préfixe (nom avant DSM/CHM)
    pairs_dict = defaultdict(dict)

    for file_path in tif_files:
        filename = file_path.stem
        filename_upper = filename.upper()

        # Détecter DSM ou CHM
        if 'DSM' in filename_upper:
            # Extraire le nom du site (tout ce qui précède DSM)
            parts = filename_upper.split('DSM')
            site_name = parts[0].rstrip('_- ')
            pairs_dict[site_name]['dsm'] = file_path
        elif 'CHM' in filename_upper:
            # Extraire le nom du site (tout ce qui précède CHM)
            parts = filename_upper.split('CHM')
            site_name = parts[0].rstrip('_- ')
            pairs_dict[site_name]['chm'] = file_path

    # Créer la liste des paires complètes
    pairs = []
    for site_name, files in pairs_dict.items():
        if 'dsm' in files and 'chm' in files:
            pairs.append({
                'site_name': site_name.lower(),
                'dsm_path': files['dsm'],
                'chm_path': files['chm']
            })
            logger.info(f"  ✓ Paire trouvée: {site_name}")
        else:
            missing = []
            if 'dsm' not in files:
                missing.append('DSM')
            if 'chm' not in files:
                missing.append('CHM')
            logger.warning(f"  ⚠ Paire incomplète pour {site_name}: manque {', '.join(missing)}")

    if not pairs:
        logger.error("Aucune paire DSM/CHM complète trouvée !")
        logger.info("Assurez-vous que vos fichiers contiennent 'DSM' et 'CHM' dans leur nom.")

    return pairs


def split_train_test(pairs: List[Dict], n_train: int = 3) -> Tuple[List[Dict], List[Dict]]:
    """
    Sépare les paires en ensembles train et test.

    Args:
        pairs: Liste des paires DSM/CHM
        n_train: Nombre de paires pour l'entraînement

    Returns:
        (train_pairs, test_pairs)
    """
    if len(pairs) < n_train + 1:
        logger.warning(f"Seulement {len(pairs)} paires trouvées. Recommandé: au moins {n_train + 1}")
        # Utiliser toutes sauf une pour train
        n_train = max(1, len(pairs) - 1)

    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]

    logger.info(f"Séparation: {len(train_pairs)} train, {len(test_pairs)} test")
    return train_pairs, test_pairs


def process_pair(
    pair: Dict,
    output_dir: Path,
    is_test: bool = False
) -> Dict[str, Path]:
    """
    Traite une paire DSM/CHM: alignement et sauvegarde.

    Args:
        pair: Dictionnaire avec 'site_name', 'dsm_path', 'chm_path'
        output_dir: Répertoire de sortie
        is_test: Si True, marque comme données de test

    Returns:
        Dictionnaire avec chemins des fichiers alignés
    """
    site_name = pair['site_name']
    subset = 'test' if is_test else 'train'
    logger.info(f"Traitement de {site_name} ({subset})")

    # Créer répertoire de sortie
    site_dir = output_dir / 'processed' / subset / site_name
    site_dir.mkdir(parents=True, exist_ok=True)

    # Vérifier alignement
    alignment = check_alignment(pair['dsm_path'], pair['chm_path'])

    if alignment['aligned']:
        logger.info(f"  ✓ DSM et CHM déjà alignés")
        # Copier les fichiers
        import shutil
        dsm_out = site_dir / f"{site_name}_DSM.tif"
        chm_out = site_dir / f"{site_name}_CHM.tif"
        shutil.copy2(pair['dsm_path'], dsm_out)
        shutil.copy2(pair['chm_path'], chm_out)
    else:
        logger.info(f"  ⚙ Alignement nécessaire: {', '.join(alignment['processing_needed'])}")
        # Aligner les rasters
        dsm_out, chm_out, info = align_rasters(
            dsm_path=pair['dsm_path'],
            chm_path=pair['chm_path'],
            output_dir=site_dir,
            prefix=site_name
        )
        logger.info(f"  ✓ Rasters alignés")

    return {
        'site_name': site_name,
        'dsm_path': Path(dsm_out),
        'chm_path': Path(chm_out),
        'subset': subset
    }


def generate_masks_for_pair(
    pair_info: Dict,
    thresholds: List[float],
    output_dir: Path
) -> List[Path]:
    """
    Génère les masques de trouées pour différents seuils.

    Args:
        pair_info: Infos de la paire (avec chm_path)
        thresholds: Liste des seuils de hauteur en mètres
        output_dir: Répertoire de sortie

    Returns:
        Liste des chemins vers les masques générés
    """
    site_name = pair_info['site_name']
    subset = pair_info['subset']
    chm_path = pair_info['chm_path']

    logger.info(f"Génération des masques pour {site_name}")

    # Créer répertoire masks
    masks_dir = output_dir / 'masks' / subset / site_name
    masks_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = []
    for threshold in thresholds:
        mask_path = masks_dir / f"{site_name}_mask_{threshold}m.tif"
        create_binary_mask(
            chm_path=chm_path,
            threshold=threshold,
            output_path=mask_path,
            below_threshold=True,
            min_gap_size=10
        )
        mask_paths.append(mask_path)
        logger.info(f"  ✓ Masque généré: seuil {threshold}m")

    return mask_paths


def create_tiles_for_pair(
    pair_info: Dict,
    mask_paths: List[Path],
    output_dir: Path,
    tile_size: int = 256,
    overlap: float = 0.2
) -> int:
    """
    Crée les tuiles pour une paire DSM/CHM.

    Args:
        pair_info: Infos de la paire
        mask_paths: Chemins vers les masques
        output_dir: Répertoire de sortie
        tile_size: Taille des tuiles en pixels
        overlap: Chevauchement entre tuiles (ratio)

    Returns:
        Nombre de tuiles créées
    """
    site_name = pair_info['site_name']
    subset = pair_info['subset']
    dsm_path = pair_info['dsm_path']

    logger.info(f"Création des tuiles pour {site_name}")

    # Créer répertoire tiles
    tiles_dir = output_dir / 'tiles' / subset
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Calculer overlap en pixels
    overlap_px = int(tile_size * overlap)

    # Créer grille de tuiles
    tile_grid = create_tile_grid(
        raster_path=dsm_path,
        tile_size=tile_size,
        overlap=overlap_px,
        min_valid_ratio=0.7
    )

    logger.info(f"  {len(tile_grid)} tuiles potentielles")

    # Extraire les tuiles
    tile_count = 0
    for i, tile_info in enumerate(tile_grid):
        tile_id = f"{site_name}_tile_{i:04d}"

        # Extraire tuile DSM
        dsm_tile_path = tiles_dir / f"{tile_id}_dsm.tif"
        extract_tile(dsm_path, tile_info['window'], dsm_tile_path)

        # Extraire tuiles des masques
        for mask_path in mask_paths:
            threshold = mask_path.stem.split('_')[-1]  # Extraire seuil du nom
            mask_tile_path = tiles_dir / f"{tile_id}_mask_{threshold}.tif"
            extract_tile(mask_path, tile_info['window'], mask_tile_path)

        tile_count += 1

    logger.info(f"  ✓ {tile_count} tuiles créées")
    return tile_count


def save_config(
    output_dir: Path,
    train_pairs: List[Dict],
    test_pairs: List[Dict],
    thresholds: List[float],
    tile_size: int
):
    """
    Sauvegarde un fichier de configuration YAML pour l'entraînement.

    Args:
        output_dir: Répertoire de sortie
        train_pairs: Paires d'entraînement
        test_pairs: Paires de test
        thresholds: Seuils utilisés
        tile_size: Taille des tuiles
    """
    config = {
        'data': {
            'tiles_dir': str(output_dir / 'tiles'),
            'train_dir': str(output_dir / 'tiles' / 'train'),
            'test_dir': str(output_dir / 'tiles' / 'test'),
            'tile_size': tile_size,
            'thresholds': thresholds,
            'n_train_sites': len(train_pairs),
            'n_test_sites': len(test_pairs),
            'train_sites': [p['site_name'] for p in train_pairs],
            'test_sites': [p['site_name'] for p in test_pairs]
        },
        'training': {
            'batch_size': 8,
            'num_workers': 4,
            'epochs': 50,
            'learning_rate': 0.001
        }
    }

    config_path = output_dir / 'data_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"✓ Configuration sauvegardée: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Préparer les données DSM/CHM pour l\'entraînement ForestGaps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Répertoire contenant les paires DSM/CHM'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Répertoire de sortie pour les données préparées'
    )
    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        default=[2.0, 5.0, 10.0],
        help='Seuils de hauteur pour la détection des trouées (en mètres)'
    )
    parser.add_argument(
        '--tile-size',
        type=int,
        default=256,
        help='Taille des tuiles en pixels (défaut: 256)'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.2,
        help='Chevauchement entre tuiles (0.0-0.5, défaut: 0.2)'
    )
    parser.add_argument(
        '--n-train',
        type=int,
        default=3,
        help='Nombre de paires pour l\'entraînement (défaut: 3)'
    )

    args = parser.parse_args()

    # Convertir en Path
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Vérifier que data_dir existe
    if not data_dir.exists():
        logger.error(f"Le répertoire {data_dir} n'existe pas")
        sys.exit(1)

    # Créer output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PRÉPARATION DES DONNÉES FORESTGAPS")
    logger.info("=" * 70)

    # 1. Trouver les paires DSM/CHM
    pairs = find_dsm_chm_pairs(data_dir)
    if not pairs:
        logger.error("Aucune paire trouvée. Vérifiez votre répertoire de données.")
        sys.exit(1)

    logger.info(f"\n{len(pairs)} paires trouvées:")
    for pair in pairs:
        logger.info(f"  - {pair['site_name']}")

    # 2. Séparer train/test
    train_pairs, test_pairs = split_train_test(pairs, args.n_train)

    logger.info("\n" + "=" * 70)
    logger.info("TRAITEMENT DES PAIRES")
    logger.info("=" * 70)

    # 3. Traiter les paires (alignement)
    processed_train = []
    for pair in train_pairs:
        processed = process_pair(pair, output_dir, is_test=False)
        processed_train.append(processed)

    processed_test = []
    for pair in test_pairs:
        processed = process_pair(pair, output_dir, is_test=True)
        processed_test.append(processed)

    # 4. Générer les masques
    logger.info("\n" + "=" * 70)
    logger.info("GÉNÉRATION DES MASQUES")
    logger.info("=" * 70)

    train_masks = {}
    for pair_info in processed_train:
        masks = generate_masks_for_pair(pair_info, args.thresholds, output_dir)
        train_masks[pair_info['site_name']] = masks

    test_masks = {}
    for pair_info in processed_test:
        masks = generate_masks_for_pair(pair_info, args.thresholds, output_dir)
        test_masks[pair_info['site_name']] = masks

    # 5. Créer les tuiles
    logger.info("\n" + "=" * 70)
    logger.info("CRÉATION DES TUILES")
    logger.info("=" * 70)

    total_train_tiles = 0
    for pair_info in processed_train:
        masks = train_masks[pair_info['site_name']]
        n_tiles = create_tiles_for_pair(
            pair_info, masks, output_dir, args.tile_size, args.overlap
        )
        total_train_tiles += n_tiles

    total_test_tiles = 0
    for pair_info in processed_test:
        masks = test_masks[pair_info['site_name']]
        n_tiles = create_tiles_for_pair(
            pair_info, masks, output_dir, args.tile_size, args.overlap
        )
        total_test_tiles += n_tiles

    # 6. Sauvegarder la configuration
    logger.info("\n" + "=" * 70)
    logger.info("SAUVEGARDE DE LA CONFIGURATION")
    logger.info("=" * 70)

    save_config(output_dir, processed_train, processed_test, args.thresholds, args.tile_size)

    # Résumé final
    logger.info("\n" + "=" * 70)
    logger.info("RÉSUMÉ")
    logger.info("=" * 70)
    logger.info(f"Sites d'entraînement: {len(processed_train)}")
    logger.info(f"Sites de test: {len(processed_test)}")
    logger.info(f"Tuiles d'entraînement: {total_train_tiles}")
    logger.info(f"Tuiles de test: {total_test_tiles}")
    logger.info(f"Seuils utilisés: {args.thresholds}")
    logger.info(f"Taille des tuiles: {args.tile_size}px")
    logger.info(f"\n✓ Préparation terminée!")
    logger.info(f"Répertoire de sortie: {output_dir}")
    logger.info(f"Configuration: {output_dir / 'data_config.yaml'}")


if __name__ == '__main__':
    main()
