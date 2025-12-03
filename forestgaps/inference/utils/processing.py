"""
Module de preprocessing et postprocessing pour l'inférence.

Ce module fournit des fonctions de haut niveau pour préparer les données DSM
pour l'inférence et post-traiter les prédictions.
"""

import numpy as np
from typing import Optional, Dict, Any

from .image_processing import (
    normalize_data,
    preprocess_for_model,
    post_process_prediction,
    apply_crf
)


def preprocess_dsm(
    data: np.ndarray,
    method: str = "min-max",
    stats: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Prétraite un DSM pour l'inférence.

    Cette fonction normalise les données DSM en utilisant la méthode spécifiée
    et les statistiques fournies (si disponibles).

    Args:
        data: Données DSM brutes (2D numpy array)
        method: Méthode de normalisation ('min-max', 'z-score', 'percentile')
        stats: Statistiques de normalisation à utiliser (optionnel)
                Devrait contenir les mêmes stats que l'entraînement

    Returns:
        Données DSM normalisées (2D numpy array)

    Example:
        >>> dsm = load_raster("path/to/dsm.tif")[0]
        >>> normalized = preprocess_dsm(dsm, method="min-max")
    """
    # Gérer les NaN
    data = np.nan_to_num(data, nan=0.0)

    # Normaliser les données
    normalized_data, used_stats = normalize_data(
        data=data,
        method=method,
        stats=stats,
        compute_stats=(stats is None)
    )

    return normalized_data


def postprocess_prediction(
    prediction: np.ndarray,
    image: Optional[np.ndarray] = None,
    method: str = "morphology",
    min_area: int = 10,
    close_kernel_size: int = 3,
    open_kernel_size: int = 3
) -> np.ndarray:
    """
    Post-traite une prédiction.

    Applique des opérations de post-traitement pour améliorer la qualité
    de la prédiction (filtrage morphologique, CRF, etc.).

    Args:
        prediction: Prédiction binaire ou probabilité (2D numpy array)
        image: Image d'entrée originale (optionnel, requis pour CRF)
        method: Méthode de post-traitement ('morphology', 'crf', 'none')
        min_area: Taille minimale des objets à conserver (pixels)
        close_kernel_size: Taille du noyau pour fermeture morphologique
        open_kernel_size: Taille du noyau pour ouverture morphologique

    Returns:
        Prédiction post-traitée (2D numpy array)

    Example:
        >>> pred = model.predict(dsm_tensor)
        >>> cleaned = postprocess_prediction(pred, method="morphology")
    """
    if method == "crf":
        if image is None:
            raise ValueError("L'image d'entrée est requise pour le CRF")
        # Appliquer CRF pour affiner les bords
        return apply_crf(
            probabilities=prediction,
            image=image
        )
    elif method == "morphology":
        # Appliquer opérations morphologiques
        return post_process_prediction(
            prediction=prediction,
            min_area=min_area,
            close_kernel_size=close_kernel_size,
            open_kernel_size=open_kernel_size
        )
    elif method == "none":
        # Pas de post-traitement
        return prediction
    else:
        raise ValueError(f"Méthode de post-traitement inconnue: {method}")


def batch_predict(
    model,
    data_batch: np.ndarray,
    device: str = "cpu",
    preprocess_fn=None,
    postprocess_fn=None
):
    """
    Exécute l'inférence sur un batch de données.

    Args:
        model: Modèle PyTorch pour l'inférence
        data_batch: Batch de données (4D: B×C×H×W)
        device: Dispositif pour l'inférence ('cpu', 'cuda')
        preprocess_fn: Fonction de preprocessing optionnelle
        postprocess_fn: Fonction de postprocessing optionnelle

    Returns:
        Prédictions pour le batch

    Example:
        >>> predictions = batch_predict(model, batch, device="cuda")
    """
    import torch

    # Prétraiter si nécessaire
    if preprocess_fn is not None:
        data_batch = preprocess_fn(data_batch)

    # Convertir en tensor si nécessaire
    if not isinstance(data_batch, torch.Tensor):
        data_batch = torch.from_numpy(data_batch).float()

    # Déplacer sur le device
    data_batch = data_batch.to(device)

    # Inférence
    model.eval()
    with torch.no_grad():
        predictions = model(data_batch)

    # Convertir en numpy
    predictions = predictions.cpu().numpy()

    # Post-traiter si nécessaire
    if postprocess_fn is not None:
        predictions = postprocess_fn(predictions)

    return predictions
