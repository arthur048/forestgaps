"""
Tests unitaires pour le registre de modèles.

Ce module contient des tests pour le système de registre permettant
d'enregistrer et de créer des modèles dynamiquement.
"""

import pytest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn

from forestgaps_dl.models.registry import ModelRegistry, register_model

# ===================================================================================================
# Classes fictives pour les tests
# ===================================================================================================

class DummyModel(nn.Module):
    """Modèle fictif pour les tests."""
    
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        return self.conv(x)

# ===================================================================================================
# Tests pour le registre de modèles
# ===================================================================================================

class TestModelRegistry:
    """Tests pour la classe ModelRegistry."""
    
    def setup_method(self):
        """Configuration avant chaque test."""
        # Réinitialiser le registre avant chaque test
        ModelRegistry._registry = {}
        
    def test_register_model(self):
        """Tester l'enregistrement d'un modèle dans le registre."""
        # Enregistrer un modèle
        ModelRegistry.register("dummy", DummyModel)
        
        # Vérifier que le modèle est enregistré
        assert "dummy" in ModelRegistry._registry
        assert ModelRegistry._registry["dummy"] == DummyModel
        
    def test_create_model(self):
        """Tester la création d'un modèle à partir du registre."""
        # Enregistrer un modèle
        ModelRegistry.register("dummy", DummyModel)
        
        # Créer une instance du modèle
        model = ModelRegistry.create("dummy", in_channels=3, out_channels=10)
        
        # Vérifier le type et les propriétés du modèle
        assert isinstance(model, DummyModel)
        assert model.conv.in_channels == 3
        assert model.conv.out_channels == 10
        
    def test_create_nonexistent_model(self):
        """Tester le comportement lors de la création d'un modèle non enregistré."""
        with pytest.raises(ValueError):
            ModelRegistry.create("nonexistent_model")
            
    def test_register_decorator(self):
        """Tester l'enregistrement de modèle via le décorateur."""
        # Définir et enregistrer un modèle avec le décorateur
        @register_model("decorated_model")
        class DecoratedModel(nn.Module):
            def __init__(self, feature_size=64):
                super().__init__()
                self.feature_size = feature_size
                self.linear = nn.Linear(feature_size, 10)
                
            def forward(self, x):
                return self.linear(x)
        
        # Vérifier que le modèle est correctement enregistré
        assert "decorated_model" in ModelRegistry._registry
        
        # Créer une instance du modèle
        model = ModelRegistry.create("decorated_model", feature_size=128)
        
        # Vérifier le type et les propriétés
        assert isinstance(model, DecoratedModel)
        assert model.feature_size == 128
        assert model.linear.in_features == 128
        
    def test_list_available_models(self):
        """Tester la récupération de la liste des modèles disponibles."""
        # Enregistrer quelques modèles
        ModelRegistry.register("model1", DummyModel)
        ModelRegistry.register("model2", DummyModel)
        
        # Récupérer la liste des modèles
        available_models = ModelRegistry.list_available_models()
        
        # Vérifier la liste
        assert "model1" in available_models
        assert "model2" in available_models
        assert len(available_models) == 2

# ===================================================================================================
# Tests pour la fonctionnalité du modèle
# ===================================================================================================

def test_model_forward_pass():
    """Tester l'exécution du modèle avec un tensor d'entrée."""
    # Enregistrer le modèle
    ModelRegistry.register("test_model", DummyModel)
    
    # Créer une instance du modèle
    model = ModelRegistry.create("test_model", in_channels=1, out_channels=1)
    
    # Créer un tensor d'entrée
    input_tensor = torch.randn(1, 1, 64, 64)
    
    # Exécuter le modèle
    output = model(input_tensor)
    
    # Vérifier la forme de sortie
    assert output.shape == (1, 1, 64, 64)

def test_model_gradient_flow():
    """Tester le flux des gradients à travers le modèle."""
    # Enregistrer le modèle
    ModelRegistry.register("gradient_model", DummyModel)
    
    # Créer une instance du modèle
    model = ModelRegistry.create("gradient_model", in_channels=1, out_channels=1)
    
    # Créer un tensor d'entrée qui nécessite des gradients
    input_tensor = torch.randn(1, 1, 64, 64, requires_grad=True)
    
    # Exécuter le modèle
    output = model(input_tensor)
    
    # Calculer une perte fictive
    loss = output.sum()
    
    # Rétropropager les gradients
    loss.backward()
    
    # Vérifier que les gradients ont bien été calculés
    assert input_tensor.grad is not None
    assert torch.any(input_tensor.grad != 0) 