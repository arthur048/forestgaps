#!/bin/bash
# Script de construction des images Docker pour ForestGaps-DL

# Définition des couleurs pour une meilleure lisibilité
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Vérifier si Docker est installé
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker n'est pas installé. Veuillez installer Docker avant de continuer.${NC}"
    exit 1
fi

# Vérifier si requirements.txt existe
if [ ! -f "requirements.txt" ]; then
    echo -e "${YELLOW}Fichier requirements.txt non trouvé. Génération automatique...${NC}"
    
    # Vérifier si le script de dépendances existe
    if [ -f "utils/dependency_analyzer.py" ]; then
        echo -e "${BLUE}Exécution de l'analyseur de dépendances...${NC}"
        python utils/dependency_analyzer.py
        
        if [ ! -f "requirements.txt" ]; then
            echo -e "${RED}Échec de la génération de requirements.txt. Veuillez le créer manuellement.${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Script d'analyse des dépendances non trouvé. Veuillez créer requirements.txt manuellement.${NC}"
        exit 1
    fi
fi

# Construction de l'image GPU
echo -e "${BLUE}Construction de l'image Docker avec support GPU...${NC}"
docker build -t forestgaps-dl:latest -f Dockerfile .

# Construction de l'image CPU
echo -e "${BLUE}Construction de l'image Docker CPU...${NC}"
docker build -t forestgaps-dl:cpu -f Dockerfile.cpu .

# Vérification des images construites
echo -e "${GREEN}Images Docker construites avec succès!${NC}"
echo -e "${BLUE}Liste des images ForestGaps-DL:${NC}"
docker images | grep forestgaps-dl

echo -e "\n${GREEN}Construction terminée. Vous pouvez maintenant utiliser:${NC}"
echo -e "  - ${YELLOW}docker run --gpus all forestgaps-dl:latest${NC} (avec GPU)"
echo -e "  - ${YELLOW}docker run forestgaps-dl:cpu${NC} (sans GPU)"
echo -e "  - ${YELLOW}docker-compose up <service>${NC} (avec docker-compose)" 