"""
Module de transformations pour la détection des trouées forestières.

Ce module fournit des classes et fonctions pour transformer les images et masques
utilisés dans les datasets de détection de trouées forestières.
"""

import logging
import random
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter

try:
    import kornia.augmentation as K
    import kornia.geometry as KG
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    logging.warning("Kornia n'est pas disponible. Les transformations GPU ne seront pas utilisables.")

# Configuration du logger
logger = logging.getLogger(__name__)


class ForestGapTransforms:
    """
    Transformations pour les données de trouées forestières.
    
    Cette classe fournit des transformations classiques pour les images et masques
    dans le cadre de la détection de trouées forestières.
    
    Attributes:
        is_train (bool): Si True, applique des transformations d'augmentation.
        prob (float): Probabilité d'appliquer chaque transformation.
        advanced_aug_prob (float): Probabilité d'appliquer les augmentations avancées.
        enable_elastic (bool): Si True, active les transformations élastiques.
    """
    
    def __init__(
        self,
        is_train: bool = True,
        prob: float = 0.5,
        advanced_aug_prob: float = 0.3,
        enable_elastic: bool = False,
        enable_flip: bool = True,
        enable_rotate: bool = True,
        enable_crop: bool = True,
        enable_noise: bool = True,
        rotate_range: Tuple[float, float] = (-30, 30),
        crop_range: Tuple[float, float] = (0.8, 1.0),
        noise_range: Tuple[float, float] = (0.0, 0.1),
        elastic_alpha: float = 50,
        elastic_sigma: float = 5
    ):
        """
        Initialise les transformations.
        
        Args:
            is_train: Si True, applique des transformations d'augmentation.
            prob: Probabilité d'appliquer chaque transformation.
            advanced_aug_prob: Probabilité d'appliquer les augmentations avancées.
            enable_elastic: Si True, active les transformations élastiques.
            enable_flip: Si True, active les retournements.
            enable_rotate: Si True, active les rotations.
            enable_crop: Si True, active les recadrages aléatoires.
            enable_noise: Si True, active l'ajout de bruit.
            rotate_range: Plage de degrés pour les rotations aléatoires.
            crop_range: Plage de ratios pour les recadrages aléatoires.
            noise_range: Plage d'intensité pour le bruit aléatoire.
            elastic_alpha: Paramètre alpha pour les transformations élastiques.
            elastic_sigma: Paramètre sigma pour les transformations élastiques.
        """
        self.is_train = is_train
        self.prob = prob
        self.advanced_aug_prob = advanced_aug_prob
        self.enable_elastic = enable_elastic
        self.enable_flip = enable_flip
        self.enable_rotate = enable_rotate
        self.enable_crop = enable_crop
        self.enable_noise = enable_noise
        self.rotate_range = rotate_range
        self.crop_range = crop_range
        self.noise_range = noise_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        
        logger.info(f"Transformations initialisées avec: is_train={is_train}, prob={prob}")
    
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applique les transformations à une paire image/masque.
        
        Args:
            image: Tensor de l'image d'entrée [C, H, W].
            mask: Tensor du masque [C, H, W].
            
        Returns:
            Tuple contenant l'image et le masque transformés.
        """
        if not self.is_train:
            return image, mask
        
        # Convertir en numpy pour les transformations
        if isinstance(image, torch.Tensor):
            image_np = image.numpy()
        else:
            image_np = image
            
        if isinstance(mask, torch.Tensor):
            mask_np = mask.numpy()
        else:
            mask_np = mask
        
        # Appliquer les transformations de base
        if self.enable_flip and random.random() < self.prob:
            # Flip horizontal
            if random.random() < 0.5:
                image_np = np.flip(image_np, axis=2)
                mask_np = np.flip(mask_np, axis=2)
            # Flip vertical
            else:
                image_np = np.flip(image_np, axis=1)
                mask_np = np.flip(mask_np, axis=1)
        
        if self.enable_rotate and random.random() < self.prob:
            # Rotation aléatoire
            angle = random.uniform(self.rotate_range[0], self.rotate_range[1])
            image_np, mask_np = self._rotate(image_np, mask_np, angle)
        
        if self.enable_crop and random.random() < self.prob:
            # Recadrage et zoom aléatoire
            zoom = random.uniform(self.crop_range[0], self.crop_range[1])
            image_np, mask_np = self._random_crop_zoom(image_np, mask_np, zoom)
        
        if self.enable_noise and random.random() < self.prob:
            # Ajout de bruit
            noise_level = random.uniform(self.noise_range[0], self.noise_range[1])
            noise = np.random.normal(0, noise_level, image_np.shape)
            image_np = image_np + noise
            # Limiter les valeurs dans [0, 1] si normalisées
            if image_np.max() <= 1.0:
                image_np = np.clip(image_np, 0, 1)
        
        # Transformations avancées avec probabilité réduite
        if self.enable_elastic and random.random() < self.advanced_aug_prob:
            # Transformation élastique
            image_np, mask_np = elastic_transform(
                image_np, mask_np,
                alpha=self.elastic_alpha,
                sigma=self.elastic_sigma
            )
        
        # Reconvertir en tensors si nécessaire
        if isinstance(image, torch.Tensor):
            image_t = torch.from_numpy(image_np)
        else:
            image_t = image_np
            
        if isinstance(mask, torch.Tensor):
            mask_t = torch.from_numpy(mask_np)
        else:
            mask_t = mask_np
        
        return image_t, mask_t
    
    def _rotate(self, image: np.ndarray, mask: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applique une rotation à l'image et au masque.
        
        Args:
            image: Image à transformer [C, H, W].
            mask: Masque à transformer [C, H, W].
            angle: Angle de rotation en degrés.
            
        Returns:
            Tuple contenant l'image et le masque transformés.
        """
        # Rotation par canal
        rotated_image = np.zeros_like(image)
        rotated_mask = np.zeros_like(mask)
        
        # Extraire les dimensions
        c, h, w = image.shape
        
        # Centre de la rotation
        center = (w / 2, h / 2)
        
        # Matrice de rotation
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Application de la rotation canal par canal
        for i in range(c):
            rotated_image[i] = cv2.warpAffine(image[i], M, (w, h), flags=cv2.INTER_LINEAR)
        
        for i in range(mask.shape[0]):
            rotated_mask[i] = cv2.warpAffine(mask[i], M, (w, h), flags=cv2.INTER_NEAREST)
        
        return rotated_image, rotated_mask
    
    def _random_crop_zoom(self, image: np.ndarray, mask: np.ndarray, zoom: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applique un recadrage aléatoire avec zoom à l'image et au masque.
        
        Args:
            image: Image à transformer [C, H, W].
            mask: Masque à transformer [C, H, W].
            zoom: Facteur de zoom (< 1 pour zoom out, > 1 pour zoom in).
            
        Returns:
            Tuple contenant l'image et le masque transformés.
        """
        c, h, w = image.shape
        
        # Calcul des nouvelles dimensions pour le zoom
        new_h = int(h * zoom)
        new_w = int(w * zoom)
        
        # Calcul des offsets pour le recadrage aléatoire
        top = random.randint(0, max(0, h - new_h))
        left = random.randint(0, max(0, w - new_w))
        
        # Recadrage
        if zoom < 1.0:  # Zoom out
            # Prendre une partie de l'image et l'agrandir
            cropped_image = np.zeros_like(image)
            cropped_mask = np.zeros_like(mask)
            
            for i in range(c):
                # Recadrage
                img_crop = image[i, top:top+new_h, left:left+new_w]
                # Redimensionnement pour revenir à la taille originale
                cropped_image[i] = cv2.resize(img_crop, (w, h), interpolation=cv2.INTER_LINEAR)
            
            for i in range(mask.shape[0]):
                mask_crop = mask[i, top:top+new_h, left:left+new_w]
                cropped_mask[i] = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)
        
        else:  # Zoom in
            # Agrandir l'image puis prendre une partie
            cropped_image = np.zeros_like(image)
            cropped_mask = np.zeros_like(mask)
            
            for i in range(c):
                # Redimensionnement
                img_resized = cv2.resize(image[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # Recadrage pour revenir à la taille originale
                if new_h > h and new_w > w:
                    top_offset = random.randint(0, new_h - h)
                    left_offset = random.randint(0, new_w - w)
                    cropped_image[i] = img_resized[top_offset:top_offset+h, left_offset:left_offset+w]
                else:
                    # Si le redimensionnement a réduit la taille (ne devrait pas arriver avec zoom > 1)
                    cropped_image[i] = cv2.resize(img_resized, (w, h), interpolation=cv2.INTER_LINEAR)
            
            for i in range(mask.shape[0]):
                mask_resized = cv2.resize(mask[i], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                if new_h > h and new_w > w:
                    top_offset = random.randint(0, new_h - h)
                    left_offset = random.randint(0, new_w - w)
                    cropped_mask[i] = mask_resized[top_offset:top_offset+h, left_offset:left_offset+w]
                else:
                    cropped_mask[i] = cv2.resize(mask_resized, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return cropped_image, cropped_mask


def elastic_transform(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 50,
    sigma: float = 5,
    random_state: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applique une transformation élastique à l'image et au masque.
    
    Args:
        image: Image à transformer [C, H, W].
        mask: Masque à transformer [C, H, W].
        alpha: Facteur d'échelle des déplacements.
        sigma: Facteur de lissage.
        random_state: État aléatoire pour la reproductibilité.
        
    Returns:
        Tuple contenant l'image et le masque transformés.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    c, h, w = image.shape
    
    # Génération des champs de déplacement
    dx = gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(h, w) * 2 - 1), sigma) * alpha
    
    # Grille de coordonnées
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    # Application de la transformation
    transformed_image = np.zeros_like(image)
    for i in range(c):
        transformed_image[i] = map_coordinates(image[i], indices, order=1).reshape(h, w)
    
    transformed_mask = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        transformed_mask[i] = map_coordinates(mask[i], indices, order=0).reshape(h, w)
    
    return transformed_image, transformed_mask


class GpuTransforms(nn.Module):
    """
    Transformations GPU pour les données de trouées forestières.
    
    Cette classe utilise Kornia pour appliquer des transformations directement sur GPU.
    
    Attributes:
        is_train (bool): Si True, applique des transformations d'augmentation.
        device (torch.device): Dispositif sur lequel appliquer les transformations.
    """
    
    def __init__(
        self,
        is_train: bool = True,
        device: Optional[torch.device] = None,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.5,
        rotate_range: Tuple[float, float] = (-30, 30),
        crop_scale_prob: float = 0.5,
        crop_scale_range: Tuple[float, float] = (0.8, 1.0),
        noise_prob: float = 0.3,
        noise_std: float = 0.1,
        motion_blur_prob: float = 0.2,
        motion_blur_kernel: int = 5,
        perspective_prob: float = 0.2,
        perspective_scale: float = 0.2,
        apply_same_transform: bool = True
    ):
        """
        Initialise les transformations GPU.
        
        Args:
            is_train: Si True, applique des transformations d'augmentation.
            device: Dispositif sur lequel appliquer les transformations.
            flip_prob: Probabilité d'appliquer un retournement.
            rotate_prob: Probabilité d'appliquer une rotation.
            rotate_range: Plage de degrés pour les rotations aléatoires.
            crop_scale_prob: Probabilité d'appliquer un recadrage avec mise à l'échelle.
            crop_scale_range: Plage de facteurs pour le recadrage.
            noise_prob: Probabilité d'ajouter du bruit.
            noise_std: Écart-type du bruit gaussien.
            motion_blur_prob: Probabilité d'appliquer un flou de mouvement.
            motion_blur_kernel: Taille du noyau pour le flou de mouvement.
            perspective_prob: Probabilité d'appliquer une transformation perspective.
            perspective_scale: Échelle de la transformation perspective.
            apply_same_transform: Si True, applique les mêmes transformations à l'image et au masque.
        """
        super().__init__()
        
        if not KORNIA_AVAILABLE:
            raise ImportError("Kornia est requis pour utiliser les transformations GPU. "
                             "Installez-le avec 'pip install kornia'.")
        
        self.is_train = is_train
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.apply_same_transform = apply_same_transform
        
        # Définition des transformations
        # Note: Pour les masques, on utilise toujours l'interpolation NEAREST
        aug_list = []
        
        if flip_prob > 0:
            aug_list.append(K.RandomHorizontalFlip(p=flip_prob))
            aug_list.append(K.RandomVerticalFlip(p=flip_prob))
        
        if rotate_prob > 0:
            aug_list.append(K.RandomRotation(
                degrees=rotate_range,
                p=rotate_prob,
                resample='bilinear',  # Pour l'image
                align_corners=True
            ))
        
        if crop_scale_prob > 0:
            aug_list.append(K.RandomResizedCrop(
                size=(None, None),  # Sera ajusté dynamiquement
                scale=crop_scale_range,
                p=crop_scale_prob,
                resample='bilinear'  # Pour l'image
            ))
        
        if noise_prob > 0:
            aug_list.append(K.RandomGaussianNoise(
                mean=0.0,
                std=noise_std,
                p=noise_prob
            ))
        
        if motion_blur_prob > 0:
            aug_list.append(K.RandomMotionBlur(
                kernel_size=motion_blur_kernel,
                angle=(-45., 45.),
                direction=(-1., 1.),
                p=motion_blur_prob
            ))
        
        if perspective_prob > 0:
            aug_list.append(K.RandomPerspective(
                distortion_scale=perspective_scale,
                p=perspective_prob
            ))
        
        # Créer les pipelines de transformation
        self.image_aug = K.AugmentationSequential(
            *aug_list,
            data_keys=["input"],
            same_on_batch=False
        )
        
        # Créer une version avec interpolation NEAREST pour les masques
        mask_aug_list = []
        for aug in aug_list:
            # Copier l'augmentation mais changer l'interpolation si approprié
            if hasattr(aug, 'resample'):
                aug_params = {k: getattr(aug, k) for k in aug.__dict__.keys() if not k.startswith('_')}
                aug_params['resample'] = 'nearest'
                mask_aug = type(aug)(**aug_params)
                mask_aug_list.append(mask_aug)
            else:
                mask_aug_list.append(aug)
        
        self.mask_aug = K.AugmentationSequential(
            *mask_aug_list,
            data_keys=["input"],
            same_on_batch=False
        )
        
        logger.info(f"Transformations GPU initialisées sur {self.device} avec {len(aug_list)} transformations")
    
    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applique les transformations à une paire image/masque.
        
        Args:
            image: Tensor de l'image d'entrée [B, C, H, W].
            mask: Tensor du masque [B, C, H, W].
            
        Returns:
            Tuple contenant l'image et le masque transformés.
        """
        if not self.is_train:
            return image, mask
        
        # Ajouter une dimension batch si nécessaire
        if image.dim() == 3:
            image = image.unsqueeze(0)
            add_batch_dim = True
        else:
            add_batch_dim = False
        
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        
        # Appliquer les transformations
        if self.apply_same_transform:
            # Générer les mêmes paramètres pour image et masque
            params = self.image_aug.forward_parameters(image.shape)
            
            # Appliquer avec les mêmes paramètres
            image = self.image_aug(image, params=params)
            mask = self.mask_aug(mask, params=params)
        else:
            # Appliquer des transformations indépendantes
            image = self.image_aug(image)
            mask = self.mask_aug(mask)
        
        # Supprimer la dimension batch si elle a été ajoutée
        if add_batch_dim:
            image = image.squeeze(0)
            mask = mask.squeeze(0)
        
        return image, mask


def create_transform_pipeline(
    transform_type: str = 'cpu',
    is_train: bool = True,
    device: Optional[torch.device] = None,
    **kwargs
) -> Callable:
    """
    Crée un pipeline de transformations adapté au type demandé.
    
    Args:
        transform_type: Type de transformation ('cpu', 'gpu', 'none').
        is_train: Si True, applique des transformations d'augmentation.
        device: Dispositif sur lequel appliquer les transformations (pour 'gpu').
        **kwargs: Arguments supplémentaires à passer au constructeur de transformations.
        
    Returns:
        Un callable qui applique les transformations à une paire image/masque.
    """
    if transform_type.lower() == 'none':
        # Pas de transformation
        return lambda image, mask: (image, mask)
    
    elif transform_type.lower() == 'cpu':
        # Transformations CPU
        return ForestGapTransforms(is_train=is_train, **kwargs)
    
    elif transform_type.lower() == 'gpu':
        # Transformations GPU
        if not KORNIA_AVAILABLE:
            logger.warning("Kornia n'est pas disponible, fallback sur transformations CPU.")
            return ForestGapTransforms(is_train=is_train, **kwargs)
        
        return GpuTransforms(is_train=is_train, device=device, **kwargs)
    
    else:
        logger.warning(f"Type de transformation '{transform_type}' non reconnu, fallback sur CPU.")
        return ForestGapTransforms(is_train=is_train, **kwargs)


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Création d'une image et d'un masque de test
    image = np.random.rand(1, 256, 256).astype(np.float32)
    mask = np.zeros((1, 256, 256), dtype=np.float32)
    mask[0, 100:150, 100:150] = 1.0  # Carré central
    
    # Conversion en tensors
    image_t = torch.from_numpy(image)
    mask_t = torch.from_numpy(mask)
    
    # Création des transformations
    transforms = ForestGapTransforms(
        is_train=True,
        prob=0.8,
        enable_elastic=True
    )
    
    # Application des transformations
    transformed_image, transformed_mask = transforms(image_t, mask_t)
    
    print(f"Image originale: {image_t.shape}, min={image_t.min()}, max={image_t.max()}")
    print(f"Image transformée: {transformed_image.shape}, min={transformed_image.min()}, max={transformed_image.max()}")
    print(f"Masque original: {mask_t.shape}, unique={torch.unique(mask_t)}")
    print(f"Masque transformé: {transformed_mask.shape}, unique={torch.unique(transformed_mask)}")
    
    # Test de la création de pipeline
    pipeline = create_transform_pipeline(
        transform_type='cpu',
        is_train=True,
        prob=0.9
    )
    
    pipeline_image, pipeline_mask = pipeline(image_t, mask_t)
    print(f"Pipeline image: {pipeline_image.shape}")
    print(f"Pipeline mask: {pipeline_mask.shape}")
    
    # Test des transformations GPU si disponibles
    if KORNIA_AVAILABLE and torch.cuda.is_available():
        device = torch.device("cuda")
        image_cuda = image_t.to(device)
        mask_cuda = mask_t.to(device)
        
        gpu_transforms = GpuTransforms(
            is_train=True,
            device=device,
            flip_prob=0.8,
            rotate_prob=0.8
        )
        
        gpu_image, gpu_mask = gpu_transforms(image_cuda, mask_cuda)
        print(f"GPU image: {gpu_image.shape}, device={gpu_image.device}")
        print(f"GPU mask: {gpu_mask.shape}, device={gpu_mask.device}")
