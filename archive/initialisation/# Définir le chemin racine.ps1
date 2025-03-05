# Définir le chemin racine
$rootPath = "G:\Mon Drive\forestgaps-dl"

# Créer le dossier racine s'il n'existe pas
New-Item -Path $rootPath -ItemType Directory -Force

# Liste des dossiers principaux à créer
$mainFolders = @("data", "models", "training", "utils", "environment", "config", "cli", "archive")

# Créer les dossiers principaux
foreach ($folder in $mainFolders) {
    New-Item -Path "$rootPath\$folder" -ItemType Directory -Force
    # Créer le fichier __init__.py dans chaque dossier, sauf pour archive
    if ($folder -ne "archive") {
        New-Item -Path "$rootPath\$folder\__init__.py" -ItemType File -Force
    }
}

# Créer le sous-dossier defaults dans config
New-Item -Path "$rootPath\config\defaults" -ItemType Directory -Force

# Fichiers dans le dossier data
$dataFiles = @("preprocessing.py", "dataset.py", "dataloader.py", "augmentation.py", "normalization.py", "storage.py")
foreach ($file in $dataFiles) {
    New-Item -Path "$rootPath\data\$file" -ItemType File -Force
}

# Fichiers dans le dossier models
$modelsFiles = @("registry.py", "unet.py", "blocks.py", "attention.py", "film.py", "normalizations.py", "droppath.py", "activations.py", "factory.py", "export.py")
foreach ($file in $modelsFiles) {
    New-Item -Path "$rootPath\models\$file" -ItemType File -Force
}

# Fichiers dans le dossier training
$trainingFiles = @("trainer.py", "metrics.py", "callbacks.py", "loss.py", "schedulers.py", "validation.py", "profiling.py")
foreach ($file in $trainingFiles) {
    New-Item -Path "$rootPath\training\$file" -ItemType File -Force
}

# Fichiers dans le dossier utils
$utilsFiles = @("visualization.py", "io.py", "logging.py", "error_handling.py", "coordinates.py", "env.py", "monitoring.py", "progress.py")
foreach ($file in $utilsFiles) {
    New-Item -Path "$rootPath\utils\$file" -ItemType File -Force
}

# Fichiers dans le dossier environment
$environmentFiles = @("base.py", "colab.py", "local.py")
foreach ($file in $environmentFiles) {
    New-Item -Path "$rootPath\environment\$file" -ItemType File -Force
}

# Fichiers dans le dossier config
$configFiles = @("base.py", "schema.py")
foreach ($file in $configFiles) {
    New-Item -Path "$rootPath\config\$file" -ItemType File -Force
}

# Fichiers dans le sous-dossier config/defaults
$configDefaultFiles = @("data.yaml", "models.yaml", "training.yaml")
foreach ($file in $configDefaultFiles) {
    New-Item -Path "$rootPath\config\defaults\$file" -ItemType File -Force
}

# Fichiers dans le dossier cli
$cliFiles = @("data.py", "train.py", "evaluate.py")
foreach ($file in $cliFiles) {
    New-Item -Path "$rootPath\cli\$file" -ItemType File -Force
}

# Fichiers à la racine
New-Item -Path "$rootPath\__init__.py" -ItemType File -Force
New-Item -Path "$rootPath\version.py" -ItemType File -Force
New-Item -Path "$rootPath\setup.py" -ItemType File -Force

# Créer les sous-dossiers dans archive pour les anciens codes
$archiveFolders = @("data_preparation", "unet_training", "notebooks")
foreach ($folder in $archiveFolders) {
    New-Item -Path "$rootPath\archive\$folder" -ItemType Directory -Force
}

# Créer des fichiers Python vides pour les scripts originaux
New-Item -Path "$rootPath\archive\data_preparation\forestgaps_dl_data_preparation.py" -ItemType File -Force
New-Item -Path "$rootPath\archive\unet_training\forestgaps_dl_u_net_training.py" -ItemType File -Force

# Créer des fichiers README.md dans chaque dossier d'archives avec tracking
New-Item -Path "$rootPath\archive\data_preparation\README.md" -ItemType File -Force -Value @"
# Préparation des données

Ce dossier contient le code original de préparation des données qui sera progressivement migré vers la nouvelle structure du package forestgaps-dl.

Fichier principal: \`forestgaps_dl_data_preparation.py\`

## État de la migration

| Fonctionnalité | Statut | Module cible | Date | Notes |
|----------------|--------|--------------|------|-------|
| Lecture des rasters | ⏳ À faire | data/preprocessing.py | | |
| Alignement des données | ⏳ À faire | data/preprocessing.py | | |
| Création de masques | ⏳ À faire | data/preprocessing.py | | |
| Normalisation | ⏳ À faire | data/normalization.py | | |
| Augmentation de données | ⏳ À faire | data/augmentation.py | | |
| Création de datasets | ⏳ À faire | data/dataset.py | | |
| Dataloaders | ⏳ À faire | data/dataloader.py | | |

**Légende:**
- ✅ Terminé
- 🔄 En cours
- ⏳ À faire
- 🔶 En attente de dépendances
"@

New-Item -Path "$rootPath\archive\unet_training\README.md" -ItemType File -Force -Value @"
# Entraînement U-Net

Ce dossier contient le code original d'entraînement du modèle U-Net qui sera progressivement migré vers la nouvelle structure du package forestgaps-dl.

Fichier principal: \`forestgaps_dl_u_net_training.py\`

## État de la migration

| Fonctionnalité | Statut | Module cible | Date | Notes |
|----------------|--------|--------------|------|-------|
| Architecture U-Net | ⏳ À faire | models/unet.py | | |
| Blocs résiduels | ⏳ À faire | models/blocks.py | | |
| Mécanismes d'attention | ⏳ À faire | models/attention.py | | |
| Fonctions de perte | ⏳ À faire | training/loss.py | | |
| Métriques | ⏳ À faire | training/metrics.py | | |
| Boucle d'entraînement | ⏳ À faire | training/trainer.py | | |
| Validation | ⏳ À faire | training/validation.py | | |
| Callbacks | ⏳ À faire | training/callbacks.py | | |
| Visualisation | ⏳ À faire | utils/visualization.py | | |

**Légende:**
- ✅ Terminé
- 🔄 En cours
- ⏳ À faire
- 🔶 En attente de dépendances
"@

New-Item -Path "$rootPath\archive\notebooks\README.md" -ItemType File -Force -Value @"
# Notebooks Colab

Ce dossier contient les notebooks Jupyter/Colab originaux du projet. Vous pouvez stocker ici vos .ipynb pour référence et documentation.

## Liste des notebooks

| Nom du notebook | Description | Statut de migration |
|-----------------|-------------|---------------------|
| | | |

**Légende:**
- ✅ Entièrement migré
- 🔄 Partiellement migré
- ⏳ À migrer
- 📝 Documentation uniquement
"@

Write-Host "Structure de dossiers et fichiers créée avec succès dans $rootPath"
Write-Host "Un dossier 'archive' a été ajouté avec deux sous-dossiers pour vos codes existants."