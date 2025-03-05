#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation du système de gestion d'environnement de ForestGaps-DL.

Ce script montre comment détecter et configurer automatiquement l'environnement
d'exécution (Colab ou local).
"""

import os
import sys
import json
from pprint import pprint

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importer les fonctions de gestion d'environnement
from forestgaps_dl.environment import (
    detect_environment,
    setup_environment,
    get_device,
    LocalEnvironment,
    ColabEnvironment
)


def main():
    """Fonction principale démontrant l'utilisation du système de gestion d'environnement."""
    print("=== Exemple d'utilisation du système de gestion d'environnement ===\n")
    
    # 1. Détecter l'environnement
    print("1. Détection de l'environnement d'exécution")
    env = detect_environment()
    
    # Afficher le type d'environnement détecté
    if isinstance(env, ColabEnvironment):
        print("✅ Environnement Google Colab détecté")
    elif isinstance(env, LocalEnvironment):
        print("✅ Environnement local détecté")
    else:
        print(f"✅ Environnement inconnu détecté: {type(env).__name__}")
    print()
    
    # 2. Configurer l'environnement
    print("2. Configuration de l'environnement")
    env = setup_environment()
    print()
    
    # 3. Obtenir des informations sur l'environnement
    print("3. Informations sur l'environnement")
    env_info = env.get_environment_info()
    for key, value in env_info.items():
        print(f"  {key}: {value}")
    print()
    
    # 4. Obtenir le dispositif à utiliser pour les calculs
    print("4. Dispositif à utiliser pour les calculs")
    device = get_device()
    print(f"✅ Dispositif détecté: {device}")
    print()
    
    # 5. Utilisation du répertoire de base
    print("5. Utilisation du répertoire de base")
    base_dir = env.get_base_dir()
    print(f"✅ Répertoire de base: {base_dir}")
    
    # Créer un sous-répertoire pour les résultats
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"✅ Répertoire pour les résultats: {results_dir}")
    
    # Sauvegarder les informations sur l'environnement
    info_file = os.path.join(results_dir, 'environment_info.json')
    with open(info_file, 'w') as f:
        json.dump(env_info, f, indent=2)
    print(f"✅ Informations sur l'environnement sauvegardées dans: {info_file}")
    print()
    
    print("=== Fin de l'exemple ===")


if __name__ == "__main__":
    main() 