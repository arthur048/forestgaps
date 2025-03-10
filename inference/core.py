"""
Classes de base pour le module d'inférence.

Ce module contient les classes principales pour exécuter l'inférence
avec des modèles préentraînés sur de nouvelles données DSM.
"""

import os
import time
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import rasterio
from tqdm import tqdm
from pathlib import Path

from forestgaps.environment import setup_environment
from forestgaps.config import ConfigurationManager, load_default_config
from forestgaps.utils.io.serialization import load_model as load_model_checkpoint
from forestgaps.models.registry import ModelRegistry

from .utils.geospatial import load_raster, save_raster, preserve_metadata
from .utils.processing import preprocess_dsm, postprocess_prediction, batch_predict
from .utils.visualization import visualize_predictions, create_comparison_figure

# Configuration du logging
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """
    Configuration pour l'inférence.
    
    Attributes:
        tiled_processing: Utiliser le traitement par tuiles pour les grandes images
        tile_size: Taille des tuiles pour le traitement par tuiles
        overlap: Chevauchement entre les tuiles (0.0 à 1.0)
        batch_size: Taille des lots pour le traitement par lots
        num_workers: Nombre de processus parallèles pour le chargement des données
        normalize_method: Méthode de normalisation ('min-max', 'z-score', etc.)
        normalize_stats: Statistiques de normalisation prédéfinies (optionnel)
        output_format: Format de sortie ('GTiff', 'PNG', etc.)
        apply_crf: Appliquer un CRF (Conditional Random Field) pour le post-traitement
        save_probability: Sauvegarder les probabilités brutes en plus des prédictions binaires
        memory_efficient: Utiliser un mode économe en mémoire pour les grandes images
        threshold_probability: Seuil de probabilité pour binariser les prédictions
    """
    tiled_processing: bool = False
    tile_size: int = 512
    overlap: float = 0.25
    batch_size: int = 1
    num_workers: int = 4
    normalize_method: str = "min-max"
    normalize_stats: Optional[Dict[str, float]] = None
    output_format: str = "GTiff"
    apply_crf: bool = False
    save_probability: bool = False
    memory_efficient: bool = False
    threshold_probability: float = 0.5


class InferenceResult:
    """
    Résultat d'une opération d'inférence.
    
    Cette classe encapsule les résultats d'une opération d'inférence,
    y compris les prédictions, les métadonnées géospatiales, et les
    informations de performance.
    """
    
    def __init__(
        self,
        prediction: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        probability: Optional[np.ndarray] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        processing_time: Optional[float] = None,
        threshold: Optional[float] = None,
        input_shape: Optional[Tuple[int, ...]] = None
    ):
        """
        Initialise un résultat d'inférence.
        
        Args:
            prediction: Prédiction binaire (masque de segmentation)
            metadata: Métadonnées géospatiales (optionnel)
            probability: Prédiction de probabilité (optionnel)
            input_path: Chemin du fichier d'entrée (optionnel)
            output_path: Chemin du fichier de sortie (optionnel)
            processing_time: Temps de traitement en secondes (optionnel)
            threshold: Seuil de hauteur utilisé (optionnel)
            input_shape: Forme de l'entrée (optionnel)
        """
        self.prediction = prediction
        self.metadata = metadata if metadata is not None else {}
        self.probability = probability
        self.input_path = input_path
        self.output_path = output_path
        self.processing_time = processing_time
        self.threshold = threshold
        self.input_shape = input_shape
        
    def save(self, output_path: Optional[str] = None, format: str = "GTiff") -> str:
        """
        Sauvegarde la prédiction dans un fichier.
        
        Args:
            output_path: Chemin où sauvegarder la prédiction (optionnel)
            format: Format de sortie ('GTiff', 'PNG', etc.)
            
        Returns:
            Chemin du fichier sauvegardé
        """
        # Utiliser le chemin fourni ou celui stocké dans l'objet
        path = output_path if output_path is not None else self.output_path
        
        if path is None:
            raise ValueError("Aucun chemin de sortie spécifié")
        
        # Sauvegarder la prédiction avec les métadonnées
        saved_path = save_raster(
            data=self.prediction,
            output_path=path,
            metadata=self.metadata,
            format=format
        )
        
        # Sauvegarder la prédiction de probabilité si disponible
        if self.probability is not None:
            prob_path = os.path.splitext(path)[0] + "_prob" + os.path.splitext(path)[1]
            save_raster(
                data=self.probability,
                output_path=prob_path,
                metadata=self.metadata,
                format=format
            )
        
        # Mettre à jour le chemin de sortie
        self.output_path = saved_path
        
        return saved_path
    
    def visualize(self, output_dir: Optional[str] = None) -> str:
        """
        Génère des visualisations de la prédiction.
        
        Args:
            output_dir: Répertoire où sauvegarder les visualisations (optionnel)
            
        Returns:
            Chemin du répertoire de visualisations
        """
        if output_dir is None:
            if self.output_path is not None:
                output_dir = os.path.dirname(self.output_path)
            else:
                raise ValueError("Aucun répertoire de sortie spécifié")
        
        # Créer le répertoire si nécessaire
        os.makedirs(output_dir, exist_ok=True)
        
        # Générer les visualisations
        visualize_predictions(
            prediction=self.prediction,
            probability=self.probability,
            output_dir=output_dir,
            base_name=os.path.basename(os.path.splitext(self.input_path)[0]) if self.input_path else "prediction",
            threshold=self.threshold
        )
        
        return output_dir


class InferenceManager:
    """
    Gestionnaire pour exécuter des inférences avec des modèles préentraînés.
    
    Cette classe fournit des méthodes pour charger un modèle préentraîné
    et l'utiliser pour faire des prédictions sur de nouvelles données DSM.
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 4
    ):
        """
        Initialise le gestionnaire d'inférence.
        
        Args:
            model_path: Chemin vers le modèle préentraîné (.pt)
            config: Configuration pour l'inférence (optionnel)
            device: Dispositif sur lequel exécuter l'inférence ('cpu', 'cuda', etc.)
            batch_size: Taille des lots pour le traitement par lots
            num_workers: Nombre de processus parallèles pour le chargement des données
        """
        self.model_path = model_path
        
        # Configurer l'environnement
        self.env = setup_environment()
        
        # Déterminer le dispositif
        if device is None:
            self.device = self.env.get_device()
        else:
            self.device = torch.device(device)
        
        # Charger la configuration
        self.config = InferenceConfig()
        
        if config is not None:
            # Mettre à jour la configuration avec les valeurs fournies
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Mettre à jour les paramètres de batch
        self.config.batch_size = batch_size
        self.config.num_workers = num_workers
        
        # Charger le modèle
        self._load_model()
        
    def _load_model(self):
        """
        Charge le modèle et prépare l'inférence.
        """
        logger.info(f"Chargement du modèle: {self.model_path}")
        
        try:
            # Charger le checkpoint
            checkpoint = load_model_checkpoint(self.model_path, device=self.device)
            
            # Extraire le nom de la classe du modèle
            model_class_name = checkpoint.get('model_class')
            
            if model_class_name is None:
                raise ValueError("Le checkpoint ne contient pas d'information sur la classe du modèle")
            
            # Trouver la classe du modèle dans le registre
            model_class = None
            for name, cls in ModelRegistry._registry.items():
                if cls.__name__ == model_class_name:
                    model_class = cls
                    break
            
            if model_class is None:
                raise ValueError(f"Classe de modèle non trouvée dans le registre: {model_class_name}")
            
            # Créer une instance du modèle
            self.model = model_class()
            
            # Charger les poids du modèle
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Déplacer le modèle sur le dispositif cible
            self.model.to(self.device)
            
            # Mettre le modèle en mode évaluation
            self.model.eval()
            
            logger.info(f"Modèle chargé avec succès: {model_class_name}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise
    
    def predict(
        self,
        dsm_path: str,
        threshold: float = 5.0,
        output_path: Optional[str] = None,
        visualize: bool = False
    ) -> InferenceResult:
        """
        Exécute l'inférence sur un fichier DSM.
        
        Args:
            dsm_path: Chemin vers le fichier DSM d'entrée
            threshold: Seuil de hauteur pour la détection des trouées (en mètres)
            output_path: Chemin pour sauvegarder la prédiction (optionnel)
            visualize: Générer des visualisations de la prédiction
            
        Returns:
            Résultat de l'inférence
        """
        logger.info(f"Exécution de l'inférence sur: {dsm_path}")
        
        start_time = time.time()
        
        # Charger le DSM
        dsm_data, metadata = load_raster(dsm_path)
        
        # Prétraiter le DSM
        preprocessed_dsm = preprocess_dsm(
            dsm_data,
            method=self.config.normalize_method,
            stats=self.config.normalize_stats
        )
        
        # Vérifier si on doit utiliser le traitement par tuiles
        if self.config.tiled_processing and (dsm_data.shape[0] > self.config.tile_size or dsm_data.shape[1] > self.config.tile_size):
            logger.info(f"Utilisation du traitement par tuiles (taille: {self.config.tile_size}, chevauchement: {self.config.overlap})")
            
            # Préparer le tenseur pour les prédictions
            prediction = np.zeros(dsm_data.shape, dtype=np.float32)
            probability = np.zeros(dsm_data.shape, dtype=np.float32)
            
            # Calculer les tuiles avec chevauchement
            height, width = dsm_data.shape
            tile_size = self.config.tile_size
            overlap_pixels = int(tile_size * self.config.overlap)
            
            # Calculer les coordonnées des tuiles
            x_starts = list(range(0, width - tile_size + 1, tile_size - overlap_pixels))
            if x_starts[-1] + tile_size < width:
                x_starts.append(width - tile_size)
                
            y_starts = list(range(0, height - tile_size + 1, tile_size - overlap_pixels))
            if y_starts[-1] + tile_size < height:
                y_starts.append(height - tile_size)
            
            # Définir la fenêtre de pondération pour fusionner les tuiles avec chevauchement
            weight_window = np.ones((tile_size, tile_size), dtype=np.float32)
            weight_window[:overlap_pixels, :] *= np.linspace(0, 1, overlap_pixels)[:, np.newaxis]
            weight_window[-overlap_pixels:, :] *= np.linspace(1, 0, overlap_pixels)[:, np.newaxis]
            weight_window[:, :overlap_pixels] *= np.linspace(0, 1, overlap_pixels)
            weight_window[:, -overlap_pixels:] *= np.linspace(1, 0, overlap_pixels)
            
            # Tracker pour accumuler les poids
            weight_tracker = np.zeros(dsm_data.shape, dtype=np.float32)
            
            # Traiter chaque tuile
            for y_start in tqdm(y_starts, desc="Traitement des tuiles (lignes)"):
                for x_start in x_starts:
                    # Extraire la tuile
                    tile = preprocessed_dsm[y_start:y_start+tile_size, x_start:x_start+tile_size]
                    
                    # Convertir en tenseur
                    tile_tensor = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0)  # Ajouter les dimensions B, C
                    tile_tensor = tile_tensor.to(self.device)
                    
                    # Créer le tenseur de seuil
                    threshold_tensor = torch.tensor([threshold], device=self.device)
                    
                    # Exécuter l'inférence
                    with torch.no_grad():
                        output = self.model(tile_tensor, threshold_tensor)
                        
                        # Appliquer sigmoid pour obtenir des probabilités
                        prob = torch.sigmoid(output)
                        
                        # Binariser la prédiction
                        pred = (prob > self.config.threshold_probability).float()
                    
                    # Convertir en numpy
                    tile_pred = pred.squeeze().cpu().numpy()
                    tile_prob = prob.squeeze().cpu().numpy()
                    
                    # Appliquer la pondération et accumuler
                    prediction[y_start:y_start+tile_size, x_start:x_start+tile_size] += tile_pred * weight_window
                    probability[y_start:y_start+tile_size, x_start:x_start+tile_size] += tile_prob * weight_window
                    weight_tracker[y_start:y_start+tile_size, x_start:x_start+tile_size] += weight_window
            
            # Normaliser les prédictions par les poids accumulés
            valid_mask = weight_tracker > 0
            prediction[valid_mask] /= weight_tracker[valid_mask]
            probability[valid_mask] /= weight_tracker[valid_mask]
            
            # Binariser à nouveau la prédiction finale
            prediction = (prediction > 0.5).astype(np.float32)
            
        else:
            # Traitement sans tuiles
            # Convertir en tenseur
            dsm_tensor = torch.from_numpy(preprocessed_dsm).float().unsqueeze(0).unsqueeze(0)  # Ajouter les dimensions B, C
            dsm_tensor = dsm_tensor.to(self.device)
            
            # Créer le tenseur de seuil
            threshold_tensor = torch.tensor([threshold], device=self.device)
            
            # Exécuter l'inférence
            with torch.no_grad():
                output = self.model(dsm_tensor, threshold_tensor)
                
                # Appliquer sigmoid pour obtenir des probabilités
                prob = torch.sigmoid(output)
                
                # Binariser la prédiction
                pred = (prob > self.config.threshold_probability).float()
            
            # Convertir en numpy
            prediction = pred.squeeze().cpu().numpy()
            probability = prob.squeeze().cpu().numpy()
        
        # Post-traitement (CRF, filtrage, etc.)
        if self.config.apply_crf:
            logger.info("Application du CRF pour le post-traitement")
            prediction = postprocess_prediction(
                prediction=prediction,
                image=dsm_data,
                method="crf"
            )
        
        # Calculer le temps de traitement
        processing_time = time.time() - start_time
        
        # Définir le chemin de sortie
        if output_path is None and dsm_path is not None:
            # Générer un chemin de sortie basé sur le chemin d'entrée
            base_dir = os.path.dirname(dsm_path)
            base_name = os.path.splitext(os.path.basename(dsm_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_prediction.tif")
        
        # Créer le résultat
        result = InferenceResult(
            prediction=prediction,
            metadata=metadata,
            probability=probability if self.config.save_probability else None,
            input_path=dsm_path,
            output_path=output_path,
            processing_time=processing_time,
            threshold=threshold,
            input_shape=dsm_data.shape
        )
        
        # Sauvegarder la prédiction si un chemin de sortie est spécifié
        if output_path is not None:
            result.save(output_path, format=self.config.output_format)
            logger.info(f"Prédiction sauvegardée dans: {output_path}")
        
        # Générer des visualisations si demandé
        if visualize:
            vis_dir = os.path.join(os.path.dirname(output_path), "visualizations")
            result.visualize(vis_dir)
            logger.info(f"Visualisations générées dans: {vis_dir}")
        
        logger.info(f"Inférence terminée en {processing_time:.2f} secondes")
        
        return result
    
    def predict_batch(
        self,
        dsm_paths: List[str],
        threshold: float = 5.0,
        output_dir: str = None,
        visualize: bool = False
    ) -> Dict[str, InferenceResult]:
        """
        Exécute l'inférence sur plusieurs fichiers DSM.
        
        Args:
            dsm_paths: Liste des chemins vers les fichiers DSM d'entrée
            threshold: Seuil de hauteur pour la détection des trouées (en mètres)
            output_dir: Répertoire pour sauvegarder les prédictions
            visualize: Générer des visualisations des prédictions
            
        Returns:
            Dictionnaire des résultats d'inférence, indexé par chemin de fichier DSM
        """
        logger.info(f"Exécution de l'inférence par lots sur {len(dsm_paths)} fichiers DSM")
        
        # Vérifier si le répertoire de sortie est spécifié
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Résultats pour chaque fichier DSM
        results = {}
        
        # Traiter chaque fichier DSM
        for dsm_path in tqdm(dsm_paths, desc="Traitement des fichiers DSM"):
            try:
                # Définir le chemin de sortie
                if output_dir is not None:
                    base_name = os.path.splitext(os.path.basename(dsm_path))[0]
                    output_path = os.path.join(output_dir, f"{base_name}_prediction.tif")
                else:
                    output_path = None
                
                # Exécuter l'inférence
                result = self.predict(
                    dsm_path=dsm_path,
                    threshold=threshold,
                    output_path=output_path,
                    visualize=visualize
                )
                
                # Stocker le résultat
                results[dsm_path] = result
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {dsm_path}: {str(e)}")
                continue
        
        logger.info(f"Inférence par lots terminée, {len(results)} fichiers traités avec succès")
        
        return results 