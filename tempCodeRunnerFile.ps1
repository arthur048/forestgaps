# Script pour configurer l'environnement de développement
# Assurez-vous d'être dans le dossier du projet: G:\Mon Drive\forestgaps-dl

# 1. Créer un environnement virtuel Python
Write-Host "Création de l'environnement virtuel Python..." -ForegroundColor Cyan
python -m venv .venv

# 2. Activer l'environnement virtuel
Write-Host "Activation de l'environnement virtuel..." -ForegroundColor Cyan
.\.venv\Scripts\Activate.ps1

# 3. Installer le projet en mode développement
Write-Host "Installation du projet en mode développement..." -ForegroundColor Cyan
pip install -e .

# 4. Installer des dépendances supplémentaires pour le développement
Write-Host "Installation des dépendances de développement..." -ForegroundColor Cyan
pip install black isort pytest pytest-cov

# 5. Vérifier l'état de Git
Write-Host "État actuel de Git:" -ForegroundColor Cyan
git status

Write-Host "`nEnvironnement configuré avec succès!" -ForegroundColor Green
Write-Host "Vous travaillez maintenant dans l'environnement virtuel Python '.venv'" -ForegroundColor Green
Write-Host "Vous êtes sur la branche Git: $(git branch --show-current)" -ForegroundColor Green

Write-Host "`nCommandes utiles:" -ForegroundColor Yellow
Write-Host "- Désactiver l'environnement virtuel: deactivate" -ForegroundColor Yellow
Write-Host "- Créer une nouvelle branche de fonctionnalité: git checkout -b feature/nom-fonctionnalite" -ForegroundColor Yellow
Write-Host "- Commiter vos changements: git add . && git commit -m 'votre message'" -ForegroundColor Yellow
Write-Host "- Pousser vos changements: git push origin develop" -ForegroundColor Yellow