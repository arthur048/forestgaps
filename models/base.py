"""
Classes de base pour les modèles de détection des trouées forestières.

Ce module définit les interfaces abstraites que tous les modèles
de segmentation doivent implémenter, assurant une cohérence entre
les différentes architectures.
"""

import abc
from typing import Dict, Any, Optional, Tuple, Union

import torch
import torch.nn as nn


class ForestGapModel(nn.Module, abc.ABC):
    """
    Classe de base abstraite pour tous les modèles de détection de trouées.
    
    Tous les modèles implémentés doivent hériter de cette classe et
    implémenter les méthodes abstraites.
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        """
        Initialise le modèle de base.
        
        Args:
            in_channels: Nombre de canaux d'entrée (par défaut: 1 pour DSM)
            out_channels: Nombre de canaux de sortie (par défaut: 1 pour masque binaire)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    @abc.abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Passage avant du modèle.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            *args: Arguments additionnels spécifiques à l'implémentation
            **kwargs: Arguments nommés additionnels spécifiques à l'implémentation
            
        Returns:
            Tenseur de sortie (segmentation) [B, out_channels, H, W]
        """
        pass
    
    @abc.abstractmethod
    def get_complexity(self) -> Dict[str, Any]:
        """
        Retourne des informations sur la complexité du modèle.
        
        Returns:
            Dictionnaire contenant des informations comme le nombre
            de paramètres, la complexité computationnelle, etc.
        """
        pass
    
    def get_num_parameters(self) -> int:
        """
        Retourne le nombre total de paramètres du modèle.
        
        Returns:
            Nombre de paramètres
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def export(self, path: str, input_shape: Tuple[int, ...], export_format: str = "onnx") -> None:
        """
        Exporte le modèle dans un format spécifique.
        
        Args:
            path: Chemin où sauvegarder le modèle exporté
            input_shape: Forme de l'entrée (B, C, H, W)
            export_format: Format d'export (onnx, torchscript)
            
        Raises:
            ValueError: Si le format d'export n'est pas supporté
        """
        if export_format.lower() == "onnx":
            self._export_onnx(path, input_shape)
        elif export_format.lower() == "torchscript":
            self._export_torchscript(path, input_shape)
        else:
            raise ValueError(f"Format d'export '{export_format}' non supporté. "
                            f"Formats disponibles: onnx, torchscript")
    
    def _export_onnx(self, path: str, input_shape: Tuple[int, ...]) -> None:
        """
        Exporte le modèle au format ONNX.
        
        Args:
            path: Chemin où sauvegarder le modèle ONNX
            input_shape: Forme de l'entrée (B, C, H, W)
        """
        self.eval()
        dummy_input = torch.randn(*input_shape, device=next(self.parameters()).device)
        
        # Vérifier l'extension du fichier
        if not path.endswith(".onnx"):
            path += ".onnx"
            
        torch.onnx.export(
            self, 
            dummy_input, 
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
    
    def _export_torchscript(self, path: str, input_shape: Tuple[int, ...]) -> None:
        """
        Exporte le modèle au format TorchScript.
        
        Args:
            path: Chemin où sauvegarder le modèle TorchScript
            input_shape: Forme de l'entrée (B, C, H, W)
        """
        self.eval()
        dummy_input = torch.randn(*input_shape, device=next(self.parameters()).device)
        
        # Tracer le modèle
        traced_script_module = torch.jit.trace(self, dummy_input)
        
        # Vérifier l'extension du fichier
        if not path.endswith(".pt") and not path.endswith(".pth"):
            path += ".pt"
            
        # Sauvegarder le modèle
        traced_script_module.save(path)


class ThresholdConditionedModel(ForestGapModel):
    """
    Classe de base pour les modèles conditionnés par un seuil de hauteur.
    
    Ces modèles prennent en entrée à la fois une image (DSM/CHM) et
    un seuil de hauteur pour la détection des trouées.
    """
    
    @abc.abstractmethod
    def forward(self, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du modèle conditionné par un seuil.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            threshold: Tenseur des seuils de hauteur [B, 1]
            
        Returns:
            Tenseur de sortie (segmentation) [B, out_channels, H, W]
        """
        pass


class UNetBaseModel(ForestGapModel):
    """
    Classe de base pour les modèles basés sur l'architecture U-Net.
    
    Cette classe fournit une structure commune et des fonctionnalités
    partagées par toutes les variantes de U-Net.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.0,
        use_sigmoid: bool = True,
        norm_layer: nn.Module = nn.BatchNorm2d,
        activation: nn.Module = nn.ReLU
    ):
        """
        Initialise le modèle U-Net de base.
        
        Args:
            in_channels: Nombre de canaux d'entrée
            out_channels: Nombre de canaux de sortie (classes)
            init_features: Nombre de caractéristiques initial
            depth: Profondeur du réseau (nombre de niveaux)
            dropout_rate: Taux de dropout
            use_sigmoid: Si True, applique une fonction sigmoid à la sortie
            norm_layer: Couche de normalisation à utiliser
            activation: Fonction d'activation à utiliser
        """
        super().__init__(in_channels, out_channels)
        
        self.init_features = init_features
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.use_sigmoid = use_sigmoid
        
        # Ces attributs seront définis par les sous-classes
        self.encoder = None
        self.bottleneck = None
        self.decoder = None
        self.final_conv = None
    
    def get_complexity(self) -> Dict[str, Any]:
        """
        Retourne des informations sur la complexité du modèle.
        
        Returns:
            Dictionnaire contenant des informations comme le nombre
            de paramètres, la taille mémoire, la profondeur, etc.
        """
        return {
            "parameters": self.get_num_parameters(),
            "depth": self.depth,
            "init_features": self.init_features,
            "model_type": self.__class__.__name__
        }


class ThresholdConditionedUNet(UNetBaseModel, ThresholdConditionedModel):
    """
    Classe de base pour les modèles U-Net conditionnés par un seuil de hauteur.
    
    Cette classe combine les fonctionnalités de UNetBaseModel et
    ThresholdConditionedModel.
    """
    
    @abc.abstractmethod
    def forward(self, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """
        Passage avant du modèle U-Net conditionné par un seuil.
        
        Args:
            x: Tenseur d'entrée [B, C, H, W]
            threshold: Tenseur des seuils de hauteur [B, 1]
            
        Returns:
            Tenseur de sortie (segmentation) [B, out_channels, H, W]
        """
        pass 