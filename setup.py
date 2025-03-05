from setuptools import setup, find_packages

setup(
    name="forestgaps-dl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "rasterio>=1.2.0",
    ],
    author="Arthur",
    author_email="arthurvdl048@email.com",
    description="Bibliothèque PyTorch pour la détection et l'analyse des trouées forestières avec le deep learning.",
    url="https://github.com/arthur048/forestgaps-dl",
)