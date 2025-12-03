# Classe Environment de base
from abc import ABC, abstractmethod
import os
import sys
from typing import Dict, Any, Optional


class Environment(ABC):
    """Classe abstraite pour la gestion d'environnement."""
    
    @classmethod
    def detect(cls):
        """
        Détecte automatiquement l'environnement d'exécution.

        Ordre de priorité:
        1. Docker (via /.dockerenv ou variables d'environnement)
        2. Google Colab (via module google.colab)
        3. Local (par défaut)

        Returns:
            Une instance de l'environnement détecté (DockerEnvironment, ColabEnvironment ou LocalEnvironment).
        """
        try:
            # Priorité 1: Vérifier Docker en premier
            # Méthode 1: Fichier /.dockerenv
            if os.path.exists("/.dockerenv"):
                from forestgaps.environment.docker import DockerEnvironment
                print("✅ Environnement Docker détecté (via /.dockerenv).")
                return DockerEnvironment()

            # Méthode 2: Variable d'environnement DOCKER_CONTAINER
            if os.environ.get("DOCKER_CONTAINER") == "1":
                from forestgaps.environment.docker import DockerEnvironment
                print("✅ Environnement Docker détecté (via DOCKER_CONTAINER).")
                return DockerEnvironment()

            # Méthode 3: Vérifier /proc/1/cgroup pour Docker/Kubernetes
            try:
                with open("/proc/1/cgroup", "r") as f:
                    content = f.read()
                    if "docker" in content or "kubepods" in content:
                        from forestgaps.environment.docker import DockerEnvironment
                        print("✅ Environnement Docker détecté (via /proc/1/cgroup).")
                        return DockerEnvironment()
            except (FileNotFoundError, PermissionError):
                pass

            # Priorité 2: Vérifier Google Colab
            # Méthode principale: Vérifier si le module 'google.colab' est disponible
            if 'google.colab' in sys.modules or os.environ.get('COLAB_GPU', '') == '1':
                from forestgaps.environment.colab import ColabEnvironment
                print("✅ Environnement Google Colab détecté.")
                return ColabEnvironment()

            # Méthode alternative: Vérifier le path système pour identifier Colab
            if '/usr/local/lib/python3' in sys.path and '/content' in os.getcwd():
                from forestgaps.environment.colab import ColabEnvironment
                print("✅ Environnement Google Colab détecté (via path système).")
                return ColabEnvironment()

            # Priorité 3: Par défaut, utiliser l'environnement local
            from forestgaps.environment.local import LocalEnvironment
            print("✅ Environnement local détecté.")
            return LocalEnvironment()

        except Exception as e:
            # Si une erreur se produit pendant la détection, utiliser l'environnement local par défaut
            print(f"⚠️ Erreur lors de la détection de l'environnement: {str(e)}")
            print("⚠️ Utilisation de l'environnement local par défaut.")
            from forestgaps.environment.local import LocalEnvironment
            return LocalEnvironment()
    
    @abstractmethod
    def setup(self):
        """
        Configure l'environnement.
        Cette méthode doit être implémentée par les sous-classes.
        """
        pass
    
    @abstractmethod
    def get_base_dir(self) -> str:
        """
        Renvoie le répertoire de base pour l'environnement.
        
        Returns:
            Chemin du répertoire de base.
        """
        pass
    
    @abstractmethod
    def mount_drive(self) -> bool:
        """
        Monte Google Drive si nécessaire.
        
        Returns:
            True si le montage a réussi ou n'était pas nécessaire, False sinon.
        """
        pass
    
    @abstractmethod
    def install_dependencies(self, packages: list = None) -> bool:
        """
        Installe les dépendances nécessaires.
        
        Args:
            packages: Liste des packages à installer. Si None, installe les dépendances par défaut.
            
        Returns:
            True si l'installation a réussi, False sinon.
        """
        pass
    
    @abstractmethod
    def setup_gpu(self) -> bool:
        """
        Configure le GPU si disponible.
        
        Returns:
            True si un GPU est disponible et a été configuré, False sinon.
        """
        pass
    
    @abstractmethod
    def get_environment_info(self) -> Dict[str, Any]:
        """
        Récupère des informations sur l'environnement d'exécution.

        Returns:
            Dictionnaire contenant des informations sur l'environnement.
        """
        pass

    def get_device(self) -> str:
        """
        Détecte et renvoie le dispositif à utiliser pour les calculs (CPU ou GPU).

        Returns:
            Le dispositif à utiliser ('cuda' ou 'cpu').
        """
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            return 'cpu'