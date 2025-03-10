# Environnement local
import os
import sys
import subprocess
import pkg_resources
import platform
from typing import Dict, Any, List, Optional
from pathlib import Path

from forestgaps.environment.base import Environment


class LocalEnvironment(Environment):
    """
    Classe pour gérer l'environnement local.
    """
    
    def __init__(self):
        """Initialise l'environnement local."""
        self.base_dir = None
        self.gpu_available = False
    
    def setup(self):
        """
        Configure l'environnement local.
        - Vérifie les dépendances
        - Configure le GPU si disponible
        """
        print("Configuration de l'environnement local...")
        
        # Obtenir le répertoire de base
        base_dir = self.get_base_dir()
        print(f"📁 Répertoire de base: {base_dir}")
        
        # Installer les dépendances
        dependencies_installed = self.install_dependencies()
        if not dependencies_installed:
            print("⚠️ Certaines dépendances n'ont pas pu être vérifiées ou installées.")
        
        # Configurer le GPU
        gpu_available = self.setup_gpu()
        if gpu_available:
            print("✅ GPU détecté et configuré.")
        else:
            print("⚠️ Aucun GPU disponible. L'exécution sera plus lente.")
        
        print("✅ Configuration de l'environnement local terminée.")
    
    def get_base_dir(self) -> str:
        """
        Renvoie le répertoire de base pour l'environnement local.
        
        Returns:
            Chemin du répertoire de base.
        """
        if self.base_dir:
            return self.base_dir
        
        # Utiliser le répertoire de travail actuel ou le répertoire du script
        if getattr(sys, 'frozen', False):
            # Si l'application est compilée (par exemple avec PyInstaller)
            script_dir = os.path.dirname(sys.executable)
        else:
            # Sinon, utiliser le répertoire du script
            script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Remonter au répertoire racine du projet (parent du dossier environment)
        self.base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
        
        return self.base_dir
    
    def mount_drive(self) -> bool:
        """
        Méthode factice pour la compatibilité avec l'interface.
        Dans l'environnement local, aucun montage n'est nécessaire.
        
        Returns:
            True car aucun montage n'est nécessaire.
        """
        return True
    
    def install_dependencies(self, packages: List[str] = None) -> bool:
        """
        Vérifie les dépendances nécessaires.
        Dans l'environnement local, on ne force pas l'installation mais on vérifie seulement.
        
        Args:
            packages: Liste des packages à vérifier. Si None, vérifie les dépendances par défaut.
            
        Returns:
            True si toutes les dépendances sont présentes, False sinon.
        """
        if packages is None:
            packages = [
                "torch",
                "torchvision",
                "numpy",
                "matplotlib",
                "rasterio",
                "geopandas",
                "PyYAML",
                "pydantic",
                "tqdm",
                "tensorboard"
            ]
        
        missing_packages = []
        for package in packages:
            package_name = package.split('==')[0].split('>=')[0]
            try:
                # Vérifier si le package est déjà installé
                pkg_resources.get_distribution(package_name)
            except pkg_resources.DistributionNotFound:
                missing_packages.append(package)
        
        if missing_packages:
            print("⚠️ Packages manquants: " + ", ".join(missing_packages))
            print("Vous pouvez les installer avec la commande:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        else:
            print("✅ Toutes les dépendances sont installées.")
            return True
    
    def setup_gpu(self) -> bool:
        """
        Configure le GPU si disponible.
        
        Returns:
            True si un GPU est disponible et a été configuré, False sinon.
        """
        try:
            import torch
            
            # Vérifier si le GPU est disponible
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                
                print(f"✅ GPU détecté: {device_name} ({device_count} dispositif(s))")
                
                # Définir les paramètres pour optimiser l'utilisation du GPU
                torch.backends.cudnn.benchmark = True
                
                self.gpu_available = True
                return True
            else:
                print("ℹ️ Aucun GPU détecté. Utilisation du CPU uniquement.")
                self.gpu_available = False
                return False
        except ImportError:
            print("❌ PyTorch n'est pas installé ou ne prend pas en charge CUDA.")
            self.gpu_available = False
            return False
        except Exception as e:
            print(f"❌ Erreur lors de la configuration du GPU: {str(e)}")
            self.gpu_available = False
            return False
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Récupère des informations sur l'environnement d'exécution.
        
        Returns:
            Dictionnaire contenant des informations sur l'environnement.
        """
        info = {
            "environment_type": "Local",
            "python_version": platform.python_version(),
            "system": platform.system(),
            "os_version": platform.version(),
            "processor": platform.processor(),
            "base_dir": self.get_base_dir(),
            "gpu_available": self.gpu_available
        }
        
        # Ajouter des informations sur le GPU si disponible
        if self.gpu_available:
            try:
                import torch
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_count"] = torch.cuda.device_count()
                info["cuda_version"] = torch.version.cuda
            except:
                pass
        
        # Ajouter des versions de packages importants
        for package in ["torch", "numpy", "rasterio", "geopandas"]:
            try:
                info[f"{package}_version"] = pkg_resources.get_distribution(package).version
            except:
                info[f"{package}_version"] = "non installé"
        
        return info