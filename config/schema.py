# Schémas de validation
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
import os


class BaseSchema(BaseModel):
    """Schéma de base pour toutes les configurations"""
    
    class Config:
        extra = "allow"  # Permet des champs supplémentaires non définis dans le schéma


class PathsSchema(BaseSchema):
    """Schéma pour les chemins de base"""
    BASE_DIR: str = Field(..., description="Répertoire de base du projet")
    DATA_DIR: str = Field(..., description="Répertoire des données")
    PROCESSED_DIR: str = Field(..., description="Répertoire des données traitées")
    MODELS_DIR: str = Field(..., description="Répertoire des modèles")
    CONFIG_DIR: str = Field(..., description="Répertoire des configurations")
    
    @validator('DATA_DIR', 'PROCESSED_DIR', 'MODELS_DIR', 'CONFIG_DIR', pre=True)
    def validate_paths(cls, v, values):
        """Valide que les chemins existent ou peuvent être créés"""
        if not os.path.exists(v):
            os.makedirs(v, exist_ok=True)
        return v


class DataSchema(BaseSchema):
    """Schéma pour la configuration des données"""
    TILE_SIZE: int = Field(256, description="Taille des tuiles en pixels")
    THRESHOLDS: List[int] = Field([10, 15, 20, 25, 30], description="Seuils de hauteur en mètres")
    TEST_SPLIT: float = Field(0.15, description="Proportion de données pour le test")
    VAL_SPLIT: float = Field(0.15, description="Proportion de données pour la validation")
    OVERLAP: float = Field(0.30, description="Chevauchement entre les tuiles lors de l'extraction")
    
    # Chemins spécifiques aux données
    DATA_EXTERNAL_TEST_DIR: Optional[str] = Field(None, description="Répertoire des données de test externes")
    TILES_DIR: Optional[str] = Field(None, description="Répertoire des tuiles")
    TRAIN_TILES_DIR: Optional[str] = Field(None, description="Répertoire des tuiles d'entraînement")
    VAL_TILES_DIR: Optional[str] = Field(None, description="Répertoire des tuiles de validation")
    TEST_TILES_DIR: Optional[str] = Field(None, description="Répertoire des tuiles de test")
    
    @validator('TEST_SPLIT', 'VAL_SPLIT')
    def validate_splits(cls, v):
        """Valide que les proportions sont entre 0 et 1"""
        if not 0 <= v <= 1:
            raise ValueError(f"La proportion doit être entre 0 et 1, reçu: {v}")
        return v
    
    @validator('OVERLAP')
    def validate_overlap(cls, v):
        """Valide que le chevauchement est entre 0 et 1"""
        if not 0 <= v < 1:
            raise ValueError(f"Le chevauchement doit être entre 0 et 1, reçu: {v}")
        return v


class ModelSchema(BaseSchema):
    """Schéma pour la configuration des modèles"""
    MODEL_TYPE: str = Field("film_cbam", description="Type de modèle ('basic', 'film', 'cbam', 'droppath', 'film_cbam', 'all')")
    IN_CHANNELS: int = Field(1, description="Nombre de canaux d'entrée")
    DROP_PATH_RATE: float = Field(0.1, description="Taux de DropPath (pour 'droppath' et 'all')")
    DROPOUT_RATE: float = Field(0.2, description="Taux de dropout pour la régularisation")
    
    @validator('MODEL_TYPE')
    def validate_model_type(cls, v):
        """Valide que le type de modèle est valide"""
        valid_types = ['basic', 'film', 'cbam', 'droppath', 'film_cbam', 'all']
        if v not in valid_types:
            raise ValueError(f"Le type de modèle doit être l'un de {valid_types}, reçu: {v}")
        return v
    
    @validator('DROP_PATH_RATE', 'DROPOUT_RATE')
    def validate_rates(cls, v):
        """Valide que les taux sont entre 0 et 1"""
        if not 0 <= v <= 1:
            raise ValueError(f"Le taux doit être entre 0 et 1, reçu: {v}")
        return v


class TrainingSchema(BaseSchema):
    """Schéma pour la configuration de l'entraînement"""
    BATCH_SIZE: int = Field(64, description="Taille des batchs pour l'entraînement")
    EPOCHS: int = Field(50, description="Nombre maximal d'époques")
    LEARNING_RATE: float = Field(0.001, description="Taux d'apprentissage initial")
    NUM_WORKERS: int = Field(8, description="Nombre de workers pour le DataLoader")
    PIN_MEMORY: bool = Field(True, description="Utiliser pin_memory pour accélérer le transfert vers GPU")
    PREFETCH_FACTOR: int = Field(50, description="Facteur de préchargement pour le DataLoader")
    AUGMENTATION: bool = Field(True, description="Activer l'augmentation des données")
    MIXUP_ALPHA: float = Field(0.2, description="Paramètre alpha pour mixup (0 = désactivé)")
    USE_AMP: bool = Field(True, description="Utiliser la précision mixte automatique")
    USE_GRADIENT_CHECKPOINTING: bool = Field(False, description="Économiser de la mémoire GPU")
    
    # Chemins spécifiques à l'entraînement
    UNET_DIR: Optional[str] = Field(None, description="Répertoire du modèle U-Net")
    CHECKPOINTS_DIR: Optional[str] = Field(None, description="Répertoire des points de contrôle")
    LOGS_DIR: Optional[str] = Field(None, description="Répertoire des logs")
    RESULTS_DIR: Optional[str] = Field(None, description="Répertoire des résultats")
    VISUALIZATIONS_DIR: Optional[str] = Field(None, description="Répertoire des visualisations")
    
    @validator('BATCH_SIZE', 'EPOCHS', 'NUM_WORKERS', 'PREFETCH_FACTOR')
    def validate_positive_integers(cls, v):
        """Valide que les valeurs sont des entiers positifs"""
        if v <= 0:
            raise ValueError(f"La valeur doit être un entier positif, reçu: {v}")
        return v
    
    @validator('LEARNING_RATE', 'MIXUP_ALPHA')
    def validate_positive_floats(cls, v):
        """Valide que les valeurs sont des flottants positifs"""
        if v < 0:
            raise ValueError(f"La valeur doit être un flottant positif, reçu: {v}")
        return v


def validate_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valide un dictionnaire de configuration selon les schémas appropriés.
    
    Args:
        config_dict: Dictionnaire de configuration à valider.
        
    Returns:
        Dictionnaire de configuration validé.
    """
    # Valider les chemins de base
    paths = PathsSchema(**config_dict)
    
    # Valider les configurations spécifiques
    data_config = DataSchema(**config_dict)
    model_config = ModelSchema(**config_dict)
    training_config = TrainingSchema(**config_dict)
    
    # Fusionner les configurations validées
    validated_config = {**paths.dict(), **data_config.dict(), **model_config.dict(), **training_config.dict()}
    
    return validated_config