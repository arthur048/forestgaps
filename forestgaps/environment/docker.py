# Environnement Docker
"""
Classe pour gérer l'environnement Docker.

Cette classe hérite de LocalEnvironment et adapte le comportement
pour un conteneur Docker.
"""

import os
import sys
import platform
from typing import Dict, Any, List
from pathlib import Path

from forestgaps.environment.local import LocalEnvironment


class DockerEnvironment(LocalEnvironment):
    """
    Classe pour gérer l'environnement Docker.

    Hérite de LocalEnvironment mais adapte certains comportements
    pour un contexte containerisé:
    - Utilise /app comme base_dir
    - Ne tente pas d'installer de dépendances (pré-installées)
    - Détecte automatiquement la présence d'un container Docker
    """

    def __init__(self):
        """Initialise l'environnement Docker."""
        super().__init__()
        self.is_docker = True
        self.base_dir = "/app"  # Répertoire standard Docker

    def setup(self):
        """
        Configure l'environnement Docker.

        Contrairement à l'environnement local, ne tente pas d'installer
        de dépendances car elles sont pré-installées dans l'image Docker.
        """
        print("Configuration de l'environnement Docker...")

        # Obtenir le répertoire de base
        base_dir = self.get_base_dir()
        print(f"📁 Répertoire de base: {base_dir}")

        # Configurer le GPU (si disponible)
        gpu_available = self.setup_gpu()
        if gpu_available:
            print("✅ GPU détecté et configuré dans le container.")
        else:
            print("ℹ️  Aucun GPU disponible (mode CPU).")

        print("✅ Configuration de l'environnement Docker terminée.")

    def get_base_dir(self) -> str:
        """
        Renvoie le répertoire de base pour l'environnement Docker.

        Dans Docker, utilise toujours /app qui est le WORKDIR standard.

        Returns:
            Chemin du répertoire de base (/app).
        """
        # Dans Docker, toujours utiliser /app
        if not self.base_dir:
            self.base_dir = "/app"

        return self.base_dir

    def mount_drive(self) -> bool:
        """
        Méthode factice pour la compatibilité avec l'interface.

        Dans l'environnement Docker, les volumes sont montés au démarrage
        du container, pas dynamiquement.

        Returns:
            True car aucun montage dynamique n'est nécessaire.
        """
        return True

    def install_dependencies(self, packages: List[str] = None) -> bool:
        """
        Dans Docker, les dépendances sont pré-installées dans l'image.

        Cette méthode ne fait que vérifier leur présence sans tenter
        de les installer.

        Args:
            packages: Liste des packages à vérifier (ignoré dans Docker).

        Returns:
            True (les dépendances sont supposées présentes).
        """
        # Dans Docker, toutes les dépendances sont pré-installées
        print("✅ Dépendances pré-installées dans l'image Docker.")
        return True

    def get_environment_info(self) -> Dict[str, Any]:
        """
        Récupère des informations sur l'environnement Docker.

        Étend les informations de LocalEnvironment avec des détails
        spécifiques au container Docker.

        Returns:
            Dictionnaire contenant des informations sur l'environnement.
        """
        # Récupérer les informations de base de LocalEnvironment
        info = super().get_environment_info()

        # Surcharger/ajouter des informations spécifiques à Docker
        info["environment_type"] = "Docker"
        info["is_docker"] = True
        info["base_dir"] = "/app"

        # Ajouter des informations Docker si disponibles
        try:
            # Vérifier si /.dockerenv existe (indicateur de container Docker)
            info["dockerenv_exists"] = os.path.exists("/.dockerenv")

            # Récupérer les variables d'environnement Docker
            docker_env_vars = {
                "DOCKER_CONTAINER": os.environ.get("DOCKER_CONTAINER", ""),
                "FORESTGAPS_ENV": os.environ.get("FORESTGAPS_ENV", ""),
                "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES", ""),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            }
            info["docker_env_vars"] = {k: v for k, v in docker_env_vars.items() if v}

            # Vérifier les volumes montés
            mounted_dirs = {}
            for dir_name in ["data", "models", "outputs", "logs"]:
                dir_path = f"/app/{dir_name}"
                if os.path.exists(dir_path):
                    mounted_dirs[dir_name] = {
                        "exists": True,
                        "writable": os.access(dir_path, os.W_OK),
                        "path": dir_path
                    }
            info["mounted_volumes"] = mounted_dirs

        except Exception as e:
            info["docker_info_error"] = str(e)

        return info

    @staticmethod
    def is_docker_environment() -> bool:
        """
        Détecte si le code s'exécute dans un container Docker.

        Méthodes de détection:
        1. Existence de /.dockerenv
        2. Variable d'environnement DOCKER_CONTAINER
        3. Présence de "docker" dans /proc/1/cgroup

        Returns:
            True si exécution dans Docker, False sinon.
        """
        # Méthode 1: Fichier /.dockerenv
        if os.path.exists("/.dockerenv"):
            return True

        # Méthode 2: Variable d'environnement
        if os.environ.get("DOCKER_CONTAINER") == "1":
            return True

        # Méthode 3: Vérifier /proc/1/cgroup (Linux uniquement)
        try:
            with open("/proc/1/cgroup", "r") as f:
                content = f.read()
                if "docker" in content or "kubepods" in content:
                    return True
        except (FileNotFoundError, PermissionError):
            pass

        return False
