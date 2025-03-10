#!/bin/bash
# Script d'exécution des conteneurs Docker pour ForestGaps-DL

# Définition des couleurs pour une meilleure lisibilité
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Afficher l'aide
show_help() {
    echo -e "${BLUE}ForestGaps-DL Docker Runner${NC}"
    echo -e "Usage: $0 [OPTIONS] COMMAND [ARGS...]"
    echo
    echo -e "Options:"
    echo -e "  -h, --help     Afficher ce message d'aide"
    echo -e "  -c, --cpu      Forcer l'utilisation de l'image CPU (sans GPU)"
    echo -e "  -g, --gpu      Forcer l'utilisation de l'image GPU (nécessite NVIDIA Docker)"
    echo -e "  -v, --volume   Spécifier des volumes supplémentaires"
    echo
    echo -e "Commandes disponibles:"
    echo -e "  train          Entraîner un modèle"
    echo -e "  predict        Effectuer des prédictions"
    echo -e "  preprocess     Prétraiter des données"
    echo -e "  evaluate       Évaluer un modèle"
    echo -e "  shell          Démarrer un shell dans le conteneur"
    echo
    echo -e "Exemples:"
    echo -e "  $0 train --config /app/config/defaults/training.yml"
    echo -e "  $0 --cpu predict --model /app/models/model.pth --input /app/data/input.tif"
    echo -e "  $0 shell"
    exit 0
}

# Initialisation des variables
USE_GPU="auto"
VOLUMES=""
COMMAND=""

# Vérifier si Docker est installé
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker n'est pas installé. Veuillez installer Docker avant de continuer.${NC}"
    exit 1
fi

# Vérifier si NVIDIA Docker est disponible
check_nvidia_docker() {
    if command -v nvidia-smi &> /dev/null && docker info | grep -q "Runtimes:.*nvidia"; then
        return 0
    else
        return 1
    fi
}

# Traiter les arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -c|--cpu)
            USE_GPU="no"
            shift
            ;;
        -g|--gpu)
            USE_GPU="yes"
            shift
            ;;
        -v|--volume)
            VOLUMES="$VOLUMES -v $2"
            shift 2
            ;;
        *)
            COMMAND="$@"
            break
            ;;
    esac
done

# Vérifier si une commande est spécifiée
if [ -z "$COMMAND" ]; then
    echo -e "${RED}Aucune commande spécifiée.${NC}"
    show_help
fi

# Déterminer si on utilise le GPU
if [ "$USE_GPU" = "auto" ]; then
    if check_nvidia_docker; then
        USE_GPU="yes"
    else
        USE_GPU="no"
    fi
fi

# Configurer la commande Docker
if [ "$USE_GPU" = "yes" ]; then
    echo -e "${GREEN}Utilisation de l'image GPU...${NC}"
    DOCKER_CMD="docker run --rm --gpus all -it $VOLUMES"
    DOCKER_IMG="forestgaps-dl:latest"
else
    echo -e "${YELLOW}Utilisation de l'image CPU...${NC}"
    DOCKER_CMD="docker run --rm -it $VOLUMES"
    DOCKER_IMG="forestgaps-dl:cpu"
fi

# Monter les volumes par défaut
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/config:/app/config"

# Gestion des commandes spécifiques
case "${COMMAND%% *}" in
    train)
        echo -e "${BLUE}Démarrage de l'entraînement...${NC}"
        DOCKER_CMD="$DOCKER_CMD -v $(pwd)/logs:/app/logs"
        ;;
    predict)
        echo -e "${BLUE}Démarrage de l'inférence...${NC}"
        DOCKER_CMD="$DOCKER_CMD -v $(pwd)/outputs:/app/outputs"
        ;;
    preprocess)
        echo -e "${BLUE}Démarrage du prétraitement...${NC}"
        ;;
    evaluate)
        echo -e "${BLUE}Démarrage de l'évaluation...${NC}"
        DOCKER_CMD="$DOCKER_CMD -v $(pwd)/outputs:/app/outputs -v $(pwd)/reports:/app/reports"
        ;;
    shell)
        echo -e "${BLUE}Démarrage d'un shell interactif...${NC}"
        COMMAND="bash"
        ;;
    *)
        ;;
esac

# Exécuter la commande
echo -e "${GREEN}Exécution:${NC} $DOCKER_CMD $DOCKER_IMG $COMMAND"
eval "$DOCKER_CMD $DOCKER_IMG $COMMAND" 