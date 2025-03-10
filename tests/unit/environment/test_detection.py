"""
Tests unitaires pour la détection et la configuration des environnements.

Ce module contient des tests pour les fonctionnalités de détection de l'environnement
d'exécution (Colab ou local) et la configuration des ressources appropriées.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

from environment import (
    detect_environment,
    setup_environment,
    get_device
)
from environment.base import Environment
from environment.colab import ColabEnvironment
from environment.local import LocalEnvironment

# ===================================================================================================
# Tests pour la détection de l'environnement
# ===================================================================================================

@patch.dict(sys.modules, {'google.colab': MagicMock()})
def test_detect_environment_colab():
    """Test que l'environnement est détecté comme Colab quand le module google.colab est disponible."""
    env = detect_environment()
    assert isinstance(env, ColabEnvironment)

@patch.dict(sys.modules, {})
@patch('environment.base.Environment.detect')
def test_detect_environment_local(mock_detect):
    """Test que l'environnement est détecté comme local quand le module google.colab n'est pas disponible."""
    mock_detect.return_value = LocalEnvironment()
    env = detect_environment()
    assert isinstance(env, LocalEnvironment)

# ===================================================================================================
# Tests pour la configuration de l'environnement
# ===================================================================================================

@patch('environment.detect_environment')
def test_setup_environment(mock_detect):
    """Test que l'environnement est correctement configuré."""
    mock_env = MagicMock()
    mock_detect.return_value = mock_env
    
    env = setup_environment()
    
    mock_detect.assert_called_once()
    mock_env.setup.assert_called_once()
    assert env == mock_env

# ===================================================================================================
# Tests pour la détection du dispositif
# ===================================================================================================

@patch('torch.cuda.is_available', return_value=True)
def test_get_device_cuda(mock_cuda):
    """Test que le dispositif est 'cuda' quand CUDA est disponible."""
    assert get_device() == 'cuda'

@patch('torch.cuda.is_available', return_value=False)
def test_get_device_cpu(mock_cuda):
    """Test que le dispositif est 'cpu' quand CUDA n'est pas disponible."""
    assert get_device() == 'cpu'

@patch('environment.get_device', side_effect=ImportError)
def test_get_device_no_torch(mock_get_device):
    """Test que le dispositif est 'cpu' quand torch n'est pas disponible."""
    assert get_device() == 'cpu'

# ===================================================================================================
# Tests pour les classes d'environnement
# ===================================================================================================

class TestEnvironmentBase:
    """Tests pour la classe de base Environment."""
    
    def test_abstract_methods(self):
        """Test que les méthodes abstraites sont bien définies."""
        abstract_methods = [
            'setup',
            'get_base_dir',
            'install_dependencies',
            'setup_gpu'
        ]
        for method in abstract_methods:
            assert hasattr(Environment, method)
            assert callable(getattr(Environment, method))

class TestColabEnvironment:
    """Tests pour la classe ColabEnvironment."""
    
    def test_initialization(self):
        """Test que l'environnement Colab est correctement initialisé."""
        env = ColabEnvironment()
        assert env.drive_mounted is False
        assert env.base_dir is None
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_gpu_detection(self, mock_cuda):
        """Test que le GPU est détecté quand il est disponible."""
        env = ColabEnvironment()
        assert env.setup_gpu() is True
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_no_gpu_detection(self, mock_cuda):
        """Test que le GPU n'est pas détecté quand il n'est pas disponible."""
        env = ColabEnvironment()
        assert env.setup_gpu() is False

class TestLocalEnvironment:
    """Tests pour la classe LocalEnvironment."""
    
    def test_initialization(self):
        """Test que l'environnement local est correctement initialisé."""
        env = LocalEnvironment()
        assert env.base_dir is None
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_gpu_detection(self, mock_cuda):
        """Test que le GPU est détecté quand il est disponible."""
        env = LocalEnvironment()
        assert env.setup_gpu() is True
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_no_gpu_detection(self, mock_cuda):
        """Test que le GPU n'est pas détecté quand il n'est pas disponible."""
        env = LocalEnvironment()
        assert env.setup_gpu() is False 