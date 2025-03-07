#!/bin/bash
# Script de test des conteneurs Docker pour ForestGaps-DL

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

# Vérifier si les images existent
check_images() {
    if ! docker image inspect forestgaps-dl:latest &> /dev/null; then
        echo -e "${RED}Image forestgaps-dl:latest non trouvée. Veuillez exécuter scripts/docker-build.sh d'abord.${NC}"
        return 1
    fi
    
    if ! docker image inspect forestgaps-dl:cpu &> /dev/null; then
        echo -e "${RED}Image forestgaps-dl:cpu non trouvée. Veuillez exécuter scripts/docker-build.sh d'abord.${NC}"
        return 1
    fi
    
    return 0
}

# Tester l'importation des modules Python
test_imports() {
    local image=$1
    local gpu_flag=$2
    
    echo -e "${BLUE}Test d'importation des modules principaux sur $image...${NC}"
    
    docker run --rm $gpu_flag $image python -c "
import torch
import torchvision
import numpy
import matplotlib
import rasterio
import pydantic
import yaml
import geopandas
import tqdm
import tensorboard
import tabulate
import pandas
import markdown
import skimage

print('PyTorch version:', torch.__version__)
print('CUDA disponible:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Dispositif CUDA:', torch.cuda.get_device_name(0))

print('\\nTous les modules ont été importés avec succès!')
"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Test d'importation réussi pour $image${NC}"
        return 0
    else
        echo -e "${RED}✗ Test d'importation échoué pour $image${NC}"
        return 1
    fi
}

# Tester l'installation du package
test_package() {
    local image=$1
    local gpu_flag=$2
    
    echo -e "${BLUE}Test d'installation du package sur $image...${NC}"
    
    docker run --rm $gpu_flag $image python -c "
import sys
import os

try:
    import forestgaps_dl
    print('ForestGaps-DL importé avec succès!')
    sys.exit(0)
except ImportError as e:
    print('Erreur d'importation:', str(e))
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Test d'installation réussi pour $image${NC}"
        return 0
    else
        echo -e "${RED}✗ Test d'installation échoué pour $image${NC}"
        return 1
    fi
}

# Exécuter tous les tests
run_tests() {
    local success=0
    local failed=0
    
    # Vérifier les images
    if ! check_images; then
        exit 1
    fi
    
    # Tester l'image GPU si NVIDIA Docker est disponible
    if command -v nvidia-smi &> /dev/null && docker info | grep -q "Runtimes:.*nvidia"; then
        echo -e "${BLUE}================================${NC}"
        echo -e "${BLUE}Tests sur l'image GPU (CUDA)${NC}"
        echo -e "${BLUE}================================${NC}"
        
        if test_imports "forestgaps-dl:latest" "--gpus all"; then
            ((success++))
        else
            ((failed++))
        fi
        
        if test_package "forestgaps-dl:latest" "--gpus all"; then
            ((success++))
        else
            ((failed++))
        fi
    else
        echo -e "${YELLOW}NVIDIA Docker non disponible, les tests GPU sont ignorés.${NC}"
    fi
    
    # Tester l'image CPU
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Tests sur l'image CPU${NC}"
    echo -e "${BLUE}================================${NC}"
    
    if test_imports "forestgaps-dl:cpu" ""; then
        ((success++))
    else
        ((failed++))
    fi
    
    if test_package "forestgaps-dl:cpu" ""; then
        ((success++))
    else
        ((failed++))
    fi
    
    # Afficher le résumé
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Résumé des tests${NC}"
    echo -e "${BLUE}================================${NC}"
    echo -e "Tests réussis: ${GREEN}$success${NC}"
    echo -e "Tests échoués: ${RED}$failed${NC}"
    
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}Tous les tests ont réussi!${NC}"
        return 0
    else
        echo -e "${RED}Certains tests ont échoué. Veuillez vérifier les erreurs ci-dessus.${NC}"
        return 1
    fi
}

# Exécuter les tests
run_tests 