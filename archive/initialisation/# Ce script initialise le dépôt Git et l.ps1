# Ce script initialise le dépôt Git et le connecte à GitHub

# Chemin racine du projet
$rootPath = "G:\Mon Drive\forestgaps-dl"

# Aller au dossier du projet
Set-Location -Path $rootPath

# Créer un fichier README.md pour le projet
$readmeContent = @"
# ForestGaps-DL

Bibliothèque PyTorch pour la détection et l'analyse des trouées forestières avec le deep learning.

## Structure du projet

- **data/**: Préparation, chargement et augmentation des données
- **models/**: Architectures de réseaux neuronaux (U-Net, etc.)
- **training/**: Logique d'entraînement, validation et métriques
- **utils/**: Fonctions utilitaires et visualisation
- **environment/**: Gestion des environnements d'exécution
- **config/**: Configuration et paramètres
- **cli/**: Interface en ligne de commande
- **archive/**: Code legacy en cours de migration

## Installation

\`\`\`bash
pip install -e .
\`\`\`

## Utilisation

Documentation en cours de développement.
"@

New-Item -Path "$rootPath\README.md" -ItemType File -Force -Value $readmeContent

# Créer un fichier .gitignore
$gitignoreContent = @"
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# Virtual environments
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# VSCode
.vscode/

# PyCharm
.idea/

# Data files
*.npy
*.npz
*.h5
*.hdf5
*.tif
*.tiff
*.tar
*.gz
*.zip
"@

New-Item -Path "$rootPath\.gitignore" -ItemType File -Force -Value $gitignoreContent

# Initialiser le dépôt Git
git init

# Ajouter tous les fichiers au staging
git add .

# Premier commit
git commit -m "Initial commit: structure du projet forestgaps-dl"

# Création de la branche develop
git branch develop
git checkout develop

# Ajouter le remote GitHub
git remote add origin https://github.com/arthur048/forestgaps-dl.git

Write-Host "Dépôt Git initialisé avec succès!"
Write-Host "Branche 'develop' créée et activée."
Write-Host ""
Write-Host "Pour pousser vers GitHub, assurez-vous d'abord que le dépôt existe sur GitHub, puis exécutez:"
Write-Host "git push -u origin main"
Write-Host "git push -u origin develop"
Write-Host ""
Write-Host "Si vous n'avez pas encore créé le dépôt GitHub, créez-le d'abord sur https://github.com/new"