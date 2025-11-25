from setuptools import setup, find_packages
import os

# Récupérer la version depuis __version__.py
import os

# On construit le chemin vers forestgaps/__version__.py
version_path = os.path.join("forestgaps", "__version__.py")

version_ns = {}
with open(version_path) as f:
    exec(f.read(), {}, version_ns)


setup(
    name="forestgaps",
    version=version_ns['__version__'],
    packages=find_packages(),
    package_data={
        '': ['*.yaml', '*.yml', '*.json'],  # Inclure les fichiers de configuration
    },
    include_package_data=True,
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "rasterio>=1.2.0",
        "pydantic>=1.8.0",
        "PyYAML>=6.0",
        "geopandas>=0.10.0",
        "tqdm>=4.60.0",
        "tensorboard>=2.8.0",
        "tabulate>=0.8.0",
        "pandas>=1.3.0",
        "markdown>=3.3.0",
        "scikit-image>=0.18.0",
    ],
    author="Arthur",
    author_email="vdlinden.arthur@gmail.com",
    description="Bibliothèque PyTorch pour la détection et l'analyse des trouées forestières avec le deep learning.",
    url="https://github.com/arthur048/forestgaps",
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)