#!/bin/bash
# Script d'installation des dépendances pour l'analyseur de dépendances

# Définition des couleurs pour une meilleure lisibilité
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Installation des dépendances pour l'analyseur de dépendances...${NC}"

# Vérifier si pip est disponible
if ! command -v pip &> /dev/null; then
    echo -e "${RED}pip n'est pas installé. Veuillez installer Python et pip avant de continuer.${NC}"
    exit 1
fi

# Installer stdlib_list pour la détection des modules de la bibliothèque standard
echo -e "${YELLOW}Installation de stdlib_list...${NC}"
pip install stdlib_list

echo -e "${GREEN}Toutes les dépendances ont été installées avec succès!${NC}"
echo -e "${BLUE}Vous pouvez maintenant exécuter l'analyseur de dépendances avec:${NC}"
echo -e "  ${YELLOW}python utils/dependency_analyzer.py${NC}" 