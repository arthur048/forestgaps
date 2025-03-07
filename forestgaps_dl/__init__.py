"""
ForestGaps-DL: Module principal pour la détection et l'analyse des trouées forestières.
"""

# Importer l'information de version depuis le package parent
try:
    from .. import __version__
except (ImportError, ValueError):
    __version__ = "0.1.1"  # Version par défaut si l'import échoue

# Pour faciliter l'importation directe des fonctions environnement
from .environment import setup_environment, detect_environment, get_device

# Modules principaux que nous exportons
__all__ = [
    "__version__",
    "evaluation", 
    "inference",
    "environment",
    "setup_environment",
    "detect_environment",
    "get_device"
] 