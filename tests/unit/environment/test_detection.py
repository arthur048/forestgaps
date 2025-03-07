"""
Tests unitaires pour la détection et la configuration des environnements.

Ce module contient des tests pour les fonctionnalités de détection de l'environnement
d'exécution (Colab ou local) et la configuration des ressources appropriées.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

from forestgaps_dl.environment.detection import (
    is_colab_environment,
    get_environment_type,
    setup_environment
)
from forestgaps_dl.environment.base import Environment, ColabEnvironment, LocalEnvironment

# ===================================================================================================
# Tests pour la détection d'environnement
# ===================================================================================================

@patch.dict(sys.modules, {'google.colab': MagicMock()})
def test_is_colab_true():
    """Tester la détection positive d'un environnement Colab."""
    assert is_colab_environment() is True

@patch.dict(sys.modules, {})
def test_is_colab_false():
    """Tester la détection négative d'un environnement Colab."""
    # Supprimer temporairement 'google.colab' de sys.modules s'il est présent
    colab_module = sys.modules.pop('google.colab', None)
    try:
        assert is_colab_environment() is False
    finally:
        # Restaurer le module s'il était présent
        if colab_module:
            sys.modules['google.colab'] = colab_module

@patch('forestgaps_dl.environment.detection.is_colab_environment', return_value=True)
def test_get_environment_type_colab(mock_is_colab):
    """Tester la récupération du type d'environnement Colab."""
    assert get_environment_type() == 'colab'

@patch('forestgaps_dl.environment.detection.is_colab_environment', return_value=False)
def test_get_environment_type_local(mock_is_colab):
    """Tester la récupération du type d'environnement local."""
    assert get_environment_type() == 'local'

# ===================================================================================================
# Tests pour la configuration de l'environnement
# ===================================================================================================

@patch('forestgaps_dl.environment.detection.is_colab_environment', return_value=True)
def test_setup_environment_colab(mock_is_colab):
    """Tester la configuration d'un environnement Colab."""
    env = setup_environment()
    assert isinstance(env, ColabEnvironment)
    assert env.name == 'colab'

@patch('forestgaps_dl.environment.detection.is_colab_environment', return_value=False)
def test_setup_environment_local(mock_is_colab):
    """Tester la configuration d'un environnement local."""
    env = setup_environment()
    assert isinstance(env, LocalEnvironment)
    assert env.name == 'local'

@patch('forestgaps_dl.environment.detection.get_environment_type', return_value='unknown')
def test_setup_environment_unknown(mock_get_type):
    """Tester le comportement avec un type d'environnement inconnu."""
    with pytest.raises(ValueError):
        setup_environment()

# ===================================================================================================
# Tests pour les classes d'environnement
# ===================================================================================================

class TestEnvironmentBase:
    """Tests pour la classe de base Environment."""
    
    def test_abstract_methods(self):
        """Tester que les méthodes abstraites sont implémentées par les sous-classes."""
        # La classe de base ne devrait pas être instanciable directement
        with pytest.raises(TypeError):
            Environment()
        
        # Les sous-classes devraient implémenter toutes les méthodes abstraites
        colab_env = ColabEnvironment()
        local_env = LocalEnvironment()
        
        # Vérifier que les méthodes principales sont implémentées
        assert hasattr(colab_env, 'setup_resources')
        assert hasattr(local_env, 'setup_resources')

class TestColabEnvironment:
    """Tests pour l'environnement Colab."""
    
    def test_initialization(self):
        """Tester l'initialisation de l'environnement Colab."""
        env = ColabEnvironment()
        assert env.name == 'colab'
        assert hasattr(env, 'config')
        
    @patch('torch.cuda.is_available', return_value=True)
    def test_gpu_detection(self, mock_cuda):
        """Tester la détection du GPU dans Colab."""
        env = ColabEnvironment()
        assert env.has_gpu is True
        
    @patch('torch.cuda.is_available', return_value=False)
    def test_no_gpu_detection(self, mock_cuda):
        """Tester la détection de l'absence de GPU dans Colab."""
        env = ColabEnvironment()
        assert env.has_gpu is False

class TestLocalEnvironment:
    """Tests pour l'environnement local."""
    
    def test_initialization(self):
        """Tester l'initialisation de l'environnement local."""
        env = LocalEnvironment()
        assert env.name == 'local'
        assert hasattr(env, 'config')
        
    @patch('torch.cuda.is_available', return_value=True)
    def test_gpu_detection(self, mock_cuda):
        """Tester la détection du GPU en environnement local."""
        env = LocalEnvironment()
        assert env.has_gpu is True
        
    @patch('torch.cuda.is_available', return_value=False)
    def test_no_gpu_detection(self, mock_cuda):
        """Tester la détection de l'absence de GPU en environnement local."""
        env = LocalEnvironment()
        assert env.has_gpu is False 