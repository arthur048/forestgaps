#!/bin/bash
# Script de nettoyage des images Docker pour ForestGaps

# Définition des couleurs pour une meilleure lisibilité
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fonction d'aide
show_help() {
    echo -e "${BLUE}ForestGaps Docker Cleanup${NC}"
    echo -e "Usage: $0 [OPTIONS]"
    echo
    echo -e "Options:"
    echo -e "  -h, --help       Afficher ce message d'aide"
    echo -e "  -a, --all        Supprimer tous les conteneurs et images Docker"
    echo -e "  -c, --containers Supprimer uniquement les conteneurs liés à ForestGaps"
    echo -e "  -i, --images     Supprimer uniquement les images liées à ForestGaps"
    echo -e "  -v, --volumes    Supprimer également les volumes Docker"
    echo
    echo -e "Par défaut, seuls les conteneurs et images de ForestGaps sont supprimés."
    exit 0
}

# Initialisation des variables
CLEAN_ALL=false
CLEAN_CONTAINERS=true
CLEAN_IMAGES=true
CLEAN_VOLUMES=false

# Vérifier si Docker est installé
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker n'est pas installé. Rien à nettoyer.${NC}"
    exit 1
fi

# Traiter les arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -a|--all)
            CLEAN_ALL=true
            shift
            ;;
        -c|--containers)
            CLEAN_IMAGES=false
            shift
            ;;
        -i|--images)
            CLEAN_CONTAINERS=false
            shift
            ;;
        -v|--volumes)
            CLEAN_VOLUMES=true
            shift
            ;;
        *)
            echo -e "${RED}Option non reconnue: $1${NC}"
            show_help
            ;;
    esac
done

# Nettoyer les conteneurs
if [ "$CLEAN_CONTAINERS" = true ]; then
    echo -e "${BLUE}Arrêt et suppression des conteneurs ForestGaps...${NC}"
    
    if [ "$CLEAN_ALL" = true ]; then
        echo -e "${YELLOW}Arrêt de tous les conteneurs Docker...${NC}"
        docker stop $(docker ps -q) 2>/dev/null || true
        echo -e "${YELLOW}Suppression de tous les conteneurs Docker...${NC}"
        docker rm $(docker ps -a -q) 2>/dev/null || true
    else
        # Arrêter et supprimer uniquement les conteneurs liés à ForestGaps
        forestgaps_containers=$(docker ps -a | grep forestgaps | awk '{print $1}')
        if [ -n "$forestgaps_containers" ]; then
            echo -e "${YELLOW}Arrêt des conteneurs ForestGaps...${NC}"
            docker stop $forestgaps_containers 2>/dev/null || true
            echo -e "${YELLOW}Suppression des conteneurs ForestGaps...${NC}"
            docker rm $forestgaps_containers 2>/dev/null || true
        else
            echo -e "${GREEN}Aucun conteneur ForestGaps trouvé.${NC}"
        fi
    fi
fi

# Nettoyer les images
if [ "$CLEAN_IMAGES" = true ]; then
    echo -e "${BLUE}Suppression des images ForestGaps...${NC}"
    
    if [ "$CLEAN_ALL" = true ]; then
        echo -e "${YELLOW}Suppression de toutes les images Docker...${NC}"
        docker rmi $(docker images -q) --force 2>/dev/null || true
    else
        # Supprimer uniquement les images liées à ForestGaps
        forestgaps_images=$(docker images | grep forestgaps | awk '{print $3}')
        if [ -n "$forestgaps_images" ]; then
            echo -e "${YELLOW}Suppression des images ForestGaps...${NC}"
            docker rmi $forestgaps_images --force 2>/dev/null || true
        else
            echo -e "${GREEN}Aucune image ForestGaps trouvée.${NC}"
        fi
    fi
fi

# Nettoyer les volumes
if [ "$CLEAN_VOLUMES" = true ]; then
    echo -e "${BLUE}Nettoyage des volumes Docker...${NC}"
    
    if [ "$CLEAN_ALL" = true ]; then
        echo -e "${YELLOW}Suppression de tous les volumes Docker...${NC}"
        docker volume prune -f
    else
        echo -e "${YELLOW}Suppression des volumes Docker non utilisés...${NC}"
        docker volume prune -f
    fi
fi

# Nettoyer les ressources non utilisées
echo -e "${BLUE}Nettoyage général des ressources Docker non utilisées...${NC}"
docker system prune -f

echo -e "${GREEN}Nettoyage Docker terminé!${NC}" 