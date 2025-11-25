"""
Fonctionnalités d'export pour les modèles de détection des trouées forestières.

Ce module fournit des fonctionnalités pour exporter les modèles entraînés
vers différents formats (ONNX, TorchScript, etc.) et pour les déployer
dans différents environnements.
"""

from .onnx import export_to_onnx, optimize_onnx_model
from .torchscript import export_to_torchscript
from .utils import trace_model, save_model_info

__all__ = [
    "export_to_onnx",
    "optimize_onnx_model",
    "export_to_torchscript",
    "trace_model",
    "save_model_info"
] 