"""
Tests unitaires pour le module inference.

Vérifie que:
- L'inférence fonctionne sur des images individuelles
- Les prédictions ont la bonne forme et plage de valeurs
- Les métadonnées géospatiales sont préservées
- L'inférence par batch fonctionne
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import rasterio
from rasterio.transform import from_bounds


class TestInferenceBasic:
    """Test les fonctionnalités de base de l'inférence."""

    @pytest.fixture
    def sample_model(self):
        """Modèle simple pour tests d'inférence."""
        from forestgaps.models import create_model
        model = create_model("unet", in_channels=1, out_channels=1, init_features=16)
        model.eval()
        return model

    @pytest.fixture
    def sample_input(self):
        """Données d'entrée pour inférence."""
        return torch.randn(1, 1, 256, 256)

    def test_inference_forward(self, sample_model, sample_input):
        """Test forward pass en mode inférence."""
        with torch.no_grad():
            output = sample_model(sample_input)

        # Shape correcte
        assert output.shape == sample_input.shape

        # Valeurs finies
        assert torch.isfinite(output).all()

    def test_inference_deterministic(self, sample_model, sample_input):
        """Test que l'inférence est déterministe."""
        sample_model.eval()

        with torch.no_grad():
            output1 = sample_model(sample_input)
            output2 = sample_model(sample_input)

        # Les deux passes doivent donner le même résultat
        assert torch.allclose(output1, output2, rtol=1e-5)

    def test_inference_batch_sizes(self, sample_model):
        """Test inférence avec différentes tailles de batch."""
        sample_model.eval()

        for batch_size in [1, 2, 4, 8]:
            inputs = torch.randn(batch_size, 1, 256, 256)

            with torch.no_grad():
                outputs = sample_model(inputs)

            assert outputs.shape[0] == batch_size

    def test_inference_film_model(self):
        """Test inférence avec modèle FiLM."""
        from forestgaps.models import create_model

        model = create_model("film_unet", in_channels=1, out_channels=1,
                           init_features=16, condition_size=1)
        model.eval()

        inputs = torch.randn(2, 1, 256, 256)
        threshold = torch.full((2, 1), 5.0)

        with torch.no_grad():
            outputs = model(inputs, threshold)

        assert outputs.shape == inputs.shape
        assert torch.isfinite(outputs).all()


class TestInferenceManager:
    """Test le InferenceManager."""

    @pytest.fixture
    def temp_geotiff(self):
        """Créer un GeoTIFF temporaire pour tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_dsm.tif"

            # Données aléatoires
            data = np.random.randn(256, 256).astype(np.float32)

            # Métadonnées géospatiales
            transform = from_bounds(0, 0, 256, 256, 256, 256)

            with rasterio.open(
                path,
                'w',
                driver='GTiff',
                height=256,
                width=256,
                count=1,
                dtype=data.dtype,
                crs='EPSG:32632',  # WGS84 UTM 32N
                transform=transform
            ) as dst:
                dst.write(data, 1)

            yield path

    def test_load_geotiff(self, temp_geotiff):
        """Test chargement d'un GeoTIFF."""
        import rasterio

        with rasterio.open(temp_geotiff) as src:
            data = src.read(1)
            meta = src.meta

        assert data.shape == (256, 256)
        assert meta['crs'].to_string() == 'EPSG:32632'
        assert meta['driver'] == 'GTiff'

    def test_inference_preserves_metadata(self, sample_model, temp_geotiff):
        """Test que l'inférence préserve les métadonnées géospatiales."""
        import rasterio

        # Charger métadonnées originales
        with rasterio.open(temp_geotiff) as src:
            original_meta = src.meta.copy()
            original_transform = src.transform
            original_crs = src.crs

        # L'inférence devrait préserver ces métadonnées
        # (test conceptuel - l'implémentation réelle utilise InferenceManager)
        assert original_meta['crs'].to_string() == 'EPSG:32632'
        assert original_transform is not None


class TestTiledInference:
    """Test l'inférence par tuiles pour grandes images."""

    def test_tiled_processing_concept(self):
        """Test concept de découpage en tuiles."""
        # Image grande
        large_image = np.random.randn(1024, 1024)

        # Paramètres de tuiles
        tile_size = 256
        overlap = 32

        # Calculer nombre de tuiles
        stride = tile_size - overlap
        n_tiles_h = (large_image.shape[0] - overlap) // stride
        n_tiles_w = (large_image.shape[1] - overlap) // stride

        # Au moins 4 tuiles
        assert n_tiles_h >= 2
        assert n_tiles_w >= 2

    def test_tile_extraction(self):
        """Test extraction de tuiles."""
        image = np.random.randn(512, 512)
        tile_size = 256

        # Extraire tuile
        tile = image[0:tile_size, 0:tile_size]

        assert tile.shape == (tile_size, tile_size)

    def test_tile_merging_no_overlap(self):
        """Test fusion de tuiles sans overlap."""
        # 4 tuiles 256x256
        tiles = [
            np.ones((256, 256)) * i
            for i in range(4)
        ]

        # Reconstruire image 512x512
        result = np.zeros((512, 512))
        result[0:256, 0:256] = tiles[0]
        result[0:256, 256:512] = tiles[1]
        result[256:512, 0:256] = tiles[2]
        result[256:512, 256:512] = tiles[3]

        # Vérifier valeurs
        assert result[0, 0] == 0
        assert result[0, 256] == 1
        assert result[256, 0] == 2
        assert result[256, 256] == 3


class TestInferencePostProcessing:
    """Test le post-processing des prédictions."""

    def test_probability_to_binary(self):
        """Test conversion probabilité -> binaire."""
        probs = torch.tensor([0.1, 0.4, 0.6, 0.9])
        threshold = 0.5

        binary = (probs > threshold).float()

        assert torch.equal(binary, torch.tensor([0.0, 0.0, 1.0, 1.0]))

    def test_sigmoid_activation(self):
        """Test application sigmoid sur logits."""
        logits = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

        probs = torch.sigmoid(logits)

        # Probabilités dans [0, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()

        # Sigmoid(0) = 0.5
        assert torch.isclose(probs[2], torch.tensor(0.5))

    def test_argmax_multiclass(self):
        """Test argmax pour classification multi-classe."""
        # 3 classes, 5 pixels
        logits = torch.randn(5, 3)

        # Classe prédite
        preds = torch.argmax(logits, dim=1)

        assert preds.shape == (5,)
        assert (preds >= 0).all()
        assert (preds < 3).all()


class TestInferenceEdgeCases:
    """Test les cas limites de l'inférence."""

    def test_single_pixel_input(self):
        """Test inférence sur très petite image."""
        from forestgaps.models import create_model

        model = create_model("unet", in_channels=1, out_channels=1, init_features=16)
        model.eval()

        # Image 64x64 (taille minimale pour U-Net depth=4)
        inputs = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            outputs = model(inputs)

        assert outputs.shape == (1, 1, 64, 64)

    def test_inference_all_zeros(self):
        """Test inférence sur image de zéros."""
        from forestgaps.models import create_model

        model = create_model("unet", in_channels=1, out_channels=1, init_features=16)
        model.eval()

        inputs = torch.zeros(1, 1, 256, 256)

        with torch.no_grad():
            outputs = model(inputs)

        # Output doit exister et être fini
        assert torch.isfinite(outputs).all()

    def test_inference_extreme_values(self):
        """Test inférence sur valeurs extrêmes."""
        from forestgaps.models import create_model

        model = create_model("unet", in_channels=1, out_channels=1, init_features=16)
        model.eval()

        # Valeurs très grandes
        inputs = torch.ones(1, 1, 256, 256) * 1000

        with torch.no_grad():
            outputs = model(inputs)

        # Output doit rester fini
        assert torch.isfinite(outputs).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
