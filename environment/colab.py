# Environnement Google Colab
import os
import sys
import subprocess
import pkg_resources
import platform
from typing import Dict, Any, List, Optional

from forestgaps.environment.base import Environment


class ColabEnvironment(Environment):
    """
    Classe pour gérer l'environnement Google Colab.
    """
    
    def __init__(self):
        """Initialise l'environnement Colab."""
        self.drive_mounted = False
        self.base_dir = None
        self.gpu_available = False
    
    def setup(self):
        """
        Configure l'environnement Google Colab.
        - Monte Google Drive
        - Installe les dépendances
        - Configure le GPU si disponible
        """
        print("Configuration de l'environnement Google Colab...")
        
        # Monter Google Drive
        drive_mounted = self.mount_drive()
        if not drive_mounted:
            print("⚠️ Impossible de monter Google Drive. Le stockage persistant ne sera pas disponible.")
        
        # Installer les dépendances
        dependencies_installed = self.install_dependencies()
        if not dependencies_installed:
            print("⚠️ Certaines dépendances n'ont pas pu être installées.")
        
        # Configurer le GPU
        gpu_available = self.setup_gpu()
        if gpu_available:
            print("✅ GPU détecté et configuré.")
        else:
            print("⚠️ Aucun GPU disponible. L'exécution sera plus lente.")
        
        print("✅ Configuration de l'environnement Colab terminée.")
    
    def get_base_dir(self) -> str:
        """
        Renvoie le répertoire de base pour l'environnement Colab.
        Si Google Drive est monté, retourne le chemin dans Drive, sinon retourne /content.
        
        Returns:
            Chemin du répertoire de base.
        """
        if self.base_dir:
            return self.base_dir
        
        if self.drive_mounted:
            # Utiliser un chemin standard dans Google Drive
            self.base_dir = '/content/drive/MyDrive/ForestGaps_DeepLearning'
            os.makedirs(self.base_dir, exist_ok=True)
        else:
            # Utiliser un chemin temporaire
            self.base_dir = '/content/forestgaps-dl'
            os.makedirs(self.base_dir, exist_ok=True)
        
        return self.base_dir
    
    def mount_drive(self) -> bool:
        """
        Monte Google Drive dans Colab.
        
        Returns:
            True si le montage a réussi, False sinon.
        """
        if self.drive_mounted:
            return True
        
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            self.drive_mounted = True
            print("✅ Google Drive monté avec succès.")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du montage de Google Drive: {str(e)}")
            self.drive_mounted = False
            return False
    
    def install_dependencies(self, packages: List[str] = None) -> bool:
        """
        Vérifie que les dépendances nécessaires sont disponibles sans réinstaller.
        Dans Colab, la plupart des packages sont déjà disponibles par défaut.
        
        Args:
            packages: Liste des packages à vérifier. Si None, vérifie les dépendances par défaut.
            
        Returns:
            True si toutes les dépendances sont disponibles, False sinon.
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
            # Extraire le nom de base du package sans version
            base_package = package.split('==')[0].split('>=')[0].split('<=')[0]
            
            try:
                # Vérifier si le package est déjà installé
                pkg_resources.get_distribution(base_package)
                print(f"✅ {base_package} est déjà installé.")
            except pkg_resources.DistributionNotFound:
                # Ajouter à la liste des packages manquants
                missing_packages.append(package)
                print(f"⚠️ {base_package} n'est pas installé.")
        
        # Installer uniquement les packages manquants
        if missing_packages:
            print(f"📦 Installation des packages manquants: {', '.join(missing_packages)}")
            try:
                # Installation silencieuse des packages manquants
                cmd = [sys.executable, "-m", "pip", "install", "-q"] + missing_packages
                subprocess.check_call(cmd)
                print("✅ Tous les packages manquants ont été installés.")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Échec de l'installation des packages: {str(e)}")
                return False
        else:
            print("✅ Toutes les dépendances sont déjà installées.")
            return True
    
    def setup_gpu(self) -> bool:
        """
        Configure le GPU dans Colab si disponible.
        
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
                torch.backends.cudnn.deterministic = False
                
                self.gpu_available = True
                return True
            else:
                print("❌ Aucun GPU détecté dans cet environnement Colab.")
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
            "environment_type": "Google Colab",
            "python_version": platform.python_version(),
            "system": platform.system(),
            "drive_mounted": self.drive_mounted,
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