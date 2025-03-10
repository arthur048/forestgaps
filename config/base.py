# Classe Config de base
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union


class Config:
    """
    Classe de configuration de base pour le projet forestgaps.
    
    Cette classe fournit les fonctionnalités de base pour charger et sauvegarder
    des configurations à partir de fichiers YAML ou JSON.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise une nouvelle instance de configuration.
        
        Args:
            config_path: Chemin optionnel vers un fichier de configuration à charger.
        """
        # Répertoire de base du projet
        self.BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Sous-répertoires principaux
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.PROCESSED_DIR = os.path.join(self.DATA_DIR, 'processed')
        self.MODELS_DIR = os.path.join(self.BASE_DIR, 'models')
        self.CONFIG_DIR = os.path.join(self.BASE_DIR, 'config')
        
        # Charger la configuration si un chemin est fourni
        if config_path:
            self.load_config(config_path)
    
    def save_config(self, filepath: Optional[str] = None, format: str = 'yaml') -> str:
        """
        Sauvegarde la configuration actuelle dans un fichier.
        
        Args:
            filepath: Chemin du fichier où sauvegarder la configuration.
                     Si None, un chemin par défaut sera utilisé.
            format: Format de sauvegarde ('yaml' ou 'json').
            
        Returns:
            Le chemin du fichier où la configuration a été sauvegardée.
        """
        if filepath is None:
            # Utiliser un chemin par défaut si aucun n'est fourni
            os.makedirs(os.path.join(self.BASE_DIR, 'config', 'user'), exist_ok=True)
            filepath = os.path.join(self.BASE_DIR, 'config', 'user', 'config.yaml' if format == 'yaml' else 'config.json')
        
        # Création d'un dictionnaire de configuration
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('__') and not callable(v)}
        
        # Conversion des chemins en chaînes de caractères
        for k, v in config_dict.items():
            if isinstance(v, Path):
                config_dict[k] = str(v)
        
        # Sauvegarde dans le format spécifié
        if format.lower() == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        else:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=4)
        
        print(f"Configuration sauvegardée dans {filepath}")
        return filepath
    
    def load_config(self, filepath: str) -> None:
        """
        Charge une configuration à partir d'un fichier.
        
        Args:
            filepath: Chemin du fichier de configuration à charger.
        """
        # Déterminer le format en fonction de l'extension
        format = 'yaml' if filepath.endswith(('.yaml', '.yml')) else 'json'
        
        # Charger le fichier dans le format approprié
        if format == 'yaml':
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        
        # Mise à jour des attributs
        for k, v in config_dict.items():
            setattr(self, k, v)
        
        print(f"Configuration chargée depuis {filepath}")
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Met à jour la configuration à partir d'un dictionnaire.
        
        Args:
            config_dict: Dictionnaire contenant les paramètres à mettre à jour.
        """
        for k, v in config_dict.items():
            setattr(self, k, v)
    
    def merge_configs(self, *config_paths: str) -> None:
        """
        Fusionne plusieurs fichiers de configuration.
        
        Args:
            *config_paths: Chemins des fichiers de configuration à fusionner.
        """
        for path in config_paths:
            self.load_config(path)
    
    def create_directories(self, directories: List[str]) -> None:
        """
        Crée les répertoires spécifiés s'ils n'existent pas.
        
        Args:
            directories: Liste des chemins de répertoires à créer.
        """
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
    
    def __str__(self) -> str:
        """Représentation sous forme de chaîne de caractères de la configuration."""
        config_str = "Configuration:\n"
        for k, v in self.__dict__.items():
            if not k.startswith('__') and not callable(v):
                config_str += f"  {k}: {v}\n"
        return config_str