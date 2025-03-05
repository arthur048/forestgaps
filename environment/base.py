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
        
        Returns:
            Une instance de l'environnement détecté (ColabEnvironment ou LocalEnvironment).
        """
        try:
            # Méthode 1: Vérifier si le module 'google.colab' est disponible
            if 'google.colab' in sys.modules:
                from forestgaps_dl.environment.colab import ColabEnvironment
                return ColabEnvironment()
            
            # Méthode 2: Essayer d'accéder à la fonction get_ipython et vérifier son origine
            try:
                import IPython
                ipython = IPython.get_ipython()
                if ipython is not None and 'google.colab' in str(ipython):
                    from forestgaps_dl.environment.colab import ColabEnvironment
                    return ColabEnvironment()
            except (ImportError, NameError):
                pass
            
            # Par défaut, utiliser l'environnement local
            from forestgaps_dl.environment.local import LocalEnvironment
            return LocalEnvironment()
        except Exception as e:
            # Si une erreur se produit pendant la détection, utiliser l'environnement local par défaut
            print(f"⚠️ Erreur lors de la détection de l'environnement: {str(e)}")
            print("⚠️ Utilisation de l'environnement local par défaut.")
            from forestgaps_dl.environment.local import LocalEnvironment
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