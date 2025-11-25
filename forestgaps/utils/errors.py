"""
Système hiérarchique d'exceptions personnalisées pour ForestGaps.

Ce module définit une hiérarchie d'exceptions spécifiques au package ForestGaps,
permettant une gestion plus précise et informative des erreurs.
"""

class ForestGapsError(Exception):
    """Classe de base pour les exceptions du workflow ForestGaps."""
    def __init__(self, message="Une erreur s'est produite dans le workflow ForestGaps"):
        self.message = message
        super().__init__(self.message)


# Hiérarchie d'exceptions pour différents modules
class DataError(ForestGapsError):
    """Erreur liée au traitement des données."""
    def __init__(self, message="Erreur lors du traitement des données"):
        super().__init__(message)


class ModelError(ForestGapsError):
    """Erreur liée aux modèles."""
    def __init__(self, message="Erreur liée au modèle"):
        super().__init__(message)


class TrainingError(ForestGapsError):
    """Erreur liée à l'entraînement."""
    def __init__(self, message="Erreur lors de l'entraînement"):
        super().__init__(message)


class ConfigError(ForestGapsError):
    """Erreur liée à la configuration."""
    def __init__(self, message="Erreur de configuration"):
        super().__init__(message)


class EnvironmentError(ForestGapsError):
    """Erreur liée à l'environnement d'exécution."""
    def __init__(self, message="Erreur d'environnement"):
        super().__init__(message)


class BenchmarkingError(ForestGapsError):
    """Erreur liée au benchmarking des modèles."""
    def __init__(self, message="Erreur lors du benchmarking des modèles"):
        super().__init__(message)


# Sous-classes spécifiques pour des erreurs plus précises
class InvalidDataFormatError(DataError):
    """Erreur de format de données invalide."""
    def __init__(self, message="Format de données invalide"):
        super().__init__(message)


class DataProcessingError(DataError):
    """Erreur lors du traitement des données."""
    def __init__(self, message="Erreur lors du traitement des données"):
        super().__init__(message)


class ModelInitializationError(ModelError):
    """Erreur lors de l'initialisation du modèle."""
    def __init__(self, message="Erreur lors de l'initialisation du modèle"):
        super().__init__(message)


class ModelLoadingError(ModelError):
    """Erreur lors du chargement du modèle."""
    def __init__(self, message="Erreur lors du chargement du modèle"):
        super().__init__(message)


class OutOfMemoryError(TrainingError):
    """Erreur de mémoire insuffisante."""
    def __init__(self, message="Mémoire insuffisante pour l'entraînement"):
        super().__init__(message)


class TrainingDivergenceError(TrainingError):
    """Erreur de divergence lors de l'entraînement."""
    def __init__(self, message="L'entraînement a divergé"):
        super().__init__(message)


# Sous-classes spécifiques pour le benchmarking
class BenchmarkConfigError(BenchmarkingError):
    """Erreur de configuration lors du benchmarking."""
    def __init__(self, message="Erreur de configuration du benchmarking"):
        super().__init__(message)


class BenchmarkExecutionError(BenchmarkingError):
    """Erreur lors de l'exécution du benchmarking."""
    def __init__(self, message="Erreur d'exécution du benchmarking"):
        super().__init__(message)


class BenchmarkReportingError(BenchmarkingError):
    """Erreur lors de la génération de rapports de benchmarking."""
    def __init__(self, message="Erreur de génération de rapport de benchmarking"):
        super().__init__(message)


class BenchmarkVisualizationError(BenchmarkingError):
    """Erreur lors de la génération de visualisations de benchmarking."""
    def __init__(self, message="Erreur de génération de visualisations de benchmarking"):
        super().__init__(message)


class ErrorHandler:
    """Gestionnaire centralisé pour les erreurs du workflow."""
    
    def __init__(self, log_file=None, verbose=True):
        """
        Initialise le gestionnaire d'erreurs.
        
        Args:
            log_file (str, optional): Chemin vers le fichier de log des erreurs.
            verbose (bool): Si True, affiche les erreurs dans la console.
        """
        self.log_file = log_file
        self.verbose = verbose
        
    def handle(self, exception, context=None):
        """
        Gère une exception en la journalisant et en affichant un message approprié.
        
        Args:
            exception (Exception): L'exception à gérer.
            context (dict, optional): Contexte supplémentaire pour le diagnostic.
            
        Returns:
            bool: True si l'exception a été gérée, False sinon.
        """
        import traceback
        from datetime import datetime
        
        # Préparer le message d'erreur
        timestamp = datetime.now().isoformat()
        error_type = type(exception).__name__
        error_message = str(exception)
        error_traceback = traceback.format_exc()
        
        # Construire le message complet
        full_message = f"[{timestamp}] {error_type}: {error_message}\n"
        if context:
            context_str = "\n".join([f"  {k}: {v}" for k, v in context.items()])
            full_message += f"Contexte:\n{context_str}\n"
        full_message += f"Traceback:\n{error_traceback}\n"
        
        # Journaliser l'erreur si un fichier de log est spécifié
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(full_message)
                    f.write("\n" + "-"*80 + "\n")
            except Exception as e:
                if self.verbose:
                    print(f"Impossible d'écrire dans le fichier de log: {str(e)}")
        
        # Afficher l'erreur si verbose est True
        if self.verbose:
            print("\n" + "="*80)
            print("ERREUR DÉTECTÉE:")
            print("-"*80)
            print(full_message)
            print("="*80 + "\n")
        
        # Indiquer que l'erreur a été gérée
        return True
