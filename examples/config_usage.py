#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation du système de configuration de ForestGaps.

Ce script montre comment charger, modifier et sauvegarder des configurations
pour le projet ForestGaps.
"""

import os
import sys
import yaml
from pathlib import Path

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forestgaps.config import (
    load_default_config,
    load_config_from_file,
    create_config_from_dict
)


def main():
    """Fonction principale démontrant l'utilisation du système de configuration."""
    print("=== Exemple d'utilisation du système de configuration ===\n")
    
    # 1. Charger la configuration par défaut
    print("1. Chargement de la configuration par défaut")
    config = load_default_config()
    print(f"Taille des tuiles : {config.TILE_SIZE}")
    print(f"Type de modèle : {config.MODEL_TYPE}")
    print(f"Taux d'apprentissage : {config.LEARNING_RATE}")
    print()
    
    # 2. Modifier la configuration
    print("2. Modification de la configuration")
    config.TILE_SIZE = 512
    config.BATCH_SIZE = 32
    config.MODEL_TYPE = "basic"
    print(f"Nouvelle taille des tuiles : {config.TILE_SIZE}")
    print(f"Nouveau type de modèle : {config.MODEL_TYPE}")
    print()
    
    # 3. Sauvegarder la configuration
    print("3. Sauvegarde de la configuration")
    os.makedirs('examples/output', exist_ok=True)
    yaml_path = config.save_config('examples/output/custom_config.yaml', format='yaml')
    json_path = config.save_config('examples/output/custom_config.json', format='json')
    print(f"Configuration sauvegardée en YAML : {yaml_path}")
    print(f"Configuration sauvegardée en JSON : {json_path}")
    print()
    
    # 4. Charger une configuration à partir d'un fichier
    print("4. Chargement d'une configuration à partir d'un fichier")
    loaded_config = load_config_from_file(yaml_path)
    print(f"Taille des tuiles chargée : {loaded_config.TILE_SIZE}")
    print(f"Type de modèle chargé : {loaded_config.MODEL_TYPE}")
    print()
    
    # 5. Créer une configuration à partir d'un dictionnaire
    print("5. Création d'une configuration à partir d'un dictionnaire")
    config_dict = {
        "TILE_SIZE": 128,
        "BATCH_SIZE": 16,
        "MODEL_TYPE": "film",
        "LEARNING_RATE": 0.0005,
        "EPOCHS": 100
    }
    custom_config = create_config_from_dict(config_dict)
    print(f"Taille des tuiles personnalisée : {custom_config.TILE_SIZE}")
    print(f"Type de modèle personnalisé : {custom_config.MODEL_TYPE}")
    print(f"Taux d'apprentissage personnalisé : {custom_config.LEARNING_RATE}")
    print()
    
    # 6. Afficher le contenu d'un fichier de configuration YAML
    print("6. Contenu du fichier de configuration YAML")
    with open(yaml_path, 'r') as f:
        yaml_content = yaml.safe_load(f)
    
    # Afficher quelques paramètres clés
    print("Paramètres clés du fichier YAML :")
    for key in ['TILE_SIZE', 'BATCH_SIZE', 'MODEL_TYPE', 'LEARNING_RATE']:
        if key in yaml_content:
            print(f"  {key}: {yaml_content[key]}")
    print()


if __name__ == "__main__":
    main() 