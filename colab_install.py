"""
Script d'installation de ForestGaps-DL pour Google Colab.

Ce script installe le package ForestGaps-DL depuis GitHub tout en évitant
la réinstallation des dépendances déjà présentes dans l'environnement Colab.
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_if_in_colab():
    """Vérifie si le script est exécuté dans Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        # Vérification alternative si le module n'est pas importable directement
        return 'google.colab' in sys.modules or os.environ.get('COLAB_GPU', '') == '1' or (
            '/usr/local/lib/python3' in sys.path and '/content' in os.getcwd()
        )

def install_package():
    """Installe ForestGaps-DL en évitant la réinstallation des dépendances."""
    print("🚀 Installation de ForestGaps-DL pour Google Colab...")
    
    # Créer un fichier temporaire de configuration pip
    pip_conf = Path("/tmp/pip.conf")
    
    with pip_conf.open("w") as f:
        f.write("[global]\n")
        f.write("no-cache-dir = true\n")
        f.write("no-warn-script-location = true\n")
        f.write("no-dependencies = true\n")  # Important : évite d'installer les dépendances

    # Configurer PIP_CONFIG_FILE pour cette session
    os.environ["PIP_CONFIG_FILE"] = str(pip_conf)
    
    # Définir les commandes d'installation
    install_command = [
        sys.executable, "-m", "pip", "install", 
        "--no-dependencies",  # Évite d'installer les dépendances
        "git+https://github.com/arthur048/forestgaps-dl.git"
    ]
    
    # Installer le package sans dépendances
    try:
        subprocess.check_call(install_command)
        print("✅ ForestGaps-DL installé avec succès (sans réinstaller les dépendances).")
        
        # Vérifier si le package est importable
        try:
            import forestgaps_dl
            print(f"✅ Module forestgaps_dl importé avec succès (version: {forestgaps_dl.__version__ if hasattr(forestgaps_dl, '__version__') else 'inconnue'}).")
        except ImportError as e:
            print(f"❌ Échec de l'importation du module: {str(e)}")
            print("⚠️ Redémarrez le runtime Colab et essayez d'importer le module à nouveau.")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Échec de l'installation: {str(e)}")
        return False
    
    return True

def check_dependencies():
    """Vérifie si les dépendances principales sont installées."""
    dependencies = [
        "torch", "torchvision", "numpy", "matplotlib", 
        "rasterio", "geopandas", "pydantic", "PyYAML",
        "tqdm", "tensorboard"
    ]
    
    missing = []
    for dep in dependencies:
        try:
            pkg_resources.get_distribution(dep)
        except pkg_resources.DistributionNotFound:
            missing.append(dep)
    
    if missing:
        print(f"⚠️ Dépendances manquantes: {', '.join(missing)}")
        print("📦 Installation des dépendances manquantes...")
        
        try:
            # Installer uniquement les dépendances manquantes
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", *missing
            ])
            print("✅ Dépendances installées.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Échec de l'installation des dépendances: {str(e)}")
            return False
    else:
        print("✅ Toutes les dépendances sont déjà installées.")
    
    return True

def main():
    """Fonction principale."""
    # Vérifier si nous sommes dans Colab
    if not check_if_in_colab():
        print("❌ Ce script est conçu pour être exécuté dans Google Colab.")
        return
    
    # Vérifier/installer les dépendances
    dependencies_ok = check_dependencies()
    if not dependencies_ok:
        print("⚠️ Problème avec les dépendances, mais nous continuons l'installation...")
    
    # Installer le package
    install_ok = install_package()
    
    if install_ok:
        print("\n✅ Installation terminée avec succès!")
        print("ℹ️ Pour utiliser ForestGaps-DL, redémarrez le runtime Colab, puis importez le module comme suit:")
        print("\nfrom forestgaps_dl.environment import setup_environment")
        print("env = setup_environment()\n")
    else:
        print("\n❌ Installation échouée.")
        print("ℹ️ Essayez d'installer manuellement avec:")
        print("\n!pip install git+https://github.com/arthur048/forestgaps-dl.git")

if __name__ == "__main__":
    main() 