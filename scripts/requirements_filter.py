#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour installer les packages disponibles et identifier ceux qui ne le sont pas.
Génère requirements.txt (packages installés) et requirements_impossible.txt (packages non installables).
"""

import subprocess
import sys
from pathlib import Path

def install_package(package):
    """Tente d'installer un package et retourne True si réussi."""
    try:
        print(f"Tentative d'installation de {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print(f"✅ {package} installé avec succès")
            return True
        else:
            print(f"❌ Échec d'installation de {package}")
            return False
    except Exception as e:
        print(f"❌ Erreur lors de l'installation de {package}: {str(e)}")
        return False

def main():
    """Fonction principale."""
    # Lire le fichier d'entrée
    input_file = Path("requirements_all.txt")  # Renommez votre fichier actuel
    if not input_file.exists():
        print(f"Le fichier {input_file} n'existe pas.")
        print("Veuillez renommer votre fichier requirements actuel en 'requirements_all.txt'")
        return 1
        
    with open(input_file, "r") as f:
        packages = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    
    # Initialiser les listes pour les packages
    installable = []
    impossible = []
    
    # Mettre à jour pip
    print("Mise à jour de pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=False)
    
    # Installer chaque package individuellement
    total = len(packages)
    for i, package in enumerate(packages, 1):
        print(f"\nTraitement de {package} ({i}/{total})...")
        if install_package(package):
            installable.append(package)
        else:
            impossible.append(package)
    
    # Écrire les résultats
    with open("requirements.txt", "w") as f:
        for package in installable:
            f.write(f"{package}\n")
            
    with open("requirements_impossible.txt", "w") as f:
        for package in impossible:
            f.write(f"{package}\n")
    
    # Afficher le résumé
    print("\n📊 Résumé:")
    print(f"Packages installés: {len(installable)}/{total}")
    print(f"Packages non installables: {len(impossible)}/{total}")
    print("\nFichiers générés:")
    print("- requirements.txt (packages installables)")
    print("- requirements_impossible.txt (packages non installables)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())