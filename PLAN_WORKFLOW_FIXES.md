# Plan de Réparation du Workflow ForestGaps

**Date:** 2025-12-03
**Objectif:** Rendre TOUT le workflow fonctionnel end-to-end

## État Actuel

### ✅ Ce qui fonctionne
1. Preprocessing fonctionnel (`scripts/prepare_training_data.py`)
   - Génère 121 tuiles depuis Plot137
   - Structure correcte: `/data/processed/tiles/train/*.tif`
   - 121 DSM + 363 masques (3 seuils)

2. Docker + GPU
   - Container opérationnel
   - GPU NVIDIA détecté et configuré

3. Environnement
   - Détection automatique (Docker/Colab/Local)
   - Config loading fonctionne

### ❌ Problèmes Identifiés

#### Critique - Bloquent le workflow
1. **Modèles UNet manquants**
   - Message: "Module unet non trouvé"
   - Impact: Benchmarking impossible avec UNet

2. **DeepLabV3Plus incomplet**
   - Méthode `get_complexity()` manquante
   - Impact: Crash à l'instanciation

3. **DataLoader retourne tuple au lieu de dict**
   - Bug dans `forestgaps/data/loaders/__init__.py:77`
   - Fix appliqué mais pas testé complètement

4. **Volume mounts Google Drive**
   - Seulement 2/14+ fichiers visibles
   - Impact: Données limitées pour training

5. **Sizes de tuiles variables**
   - Erreur: "Trying to resize storage that is not resizable"
   - Impact: DataLoader crash

#### Non-bloquant mais important
6. **CLI training_cli.py**
   - Erreur syntax ligne 15: `forestgaps.configManager`
   - Devrait être `ConfigManager`

7. **Script benchmark_quick_test.py**
   - Ne reconnaît pas les modèles du registry
   - Liste hardcodée de modèles obsolète

## Plan d'Action

### Phase 1: Fixes Critiques (Priorité 1)
- [ ] Fix import dans training_cli.py
- [ ] Investiguer module UNet manquant
- [ ] Implémenter get_complexity() pour DeepLabV3Plus
- [ ] Vérifier toutes les tuiles font 256x256
- [ ] Tester DataLoader dict fix

### Phase 2: Training Fonctionnel (Priorité 1)
- [ ] Créer script de training minimal qui marche
- [ ] Tester avec DeepLabV3Plus si UNet pas dispo
- [ ] Valider que le training produit un modèle
- [ ] Sauvegarder le modèle entraîné

### Phase 3: TensorBoard (Priorité 2)
- [ ] Vérifier que TensorBoard container tourne
- [ ] Tester accès via localhost:6006
- [ ] Valider que les logs sont écrits
- [ ] Valider que les graphs s'affichent

### Phase 4: Volume Mounts (Priorité 2)
- [ ] Documenter le problème Google Drive
- [ ] Tester avec chemin sur C: si possible
- [ ] Ou documenter workaround (docker cp)

### Phase 5: Workflow Complet (Priorité 1)
- [ ] Test end-to-end preprocessing → training → evaluation
- [ ] Valider tous les outputs générés
- [ ] Vérifier structure des dossiers
- [ ] Documenter le workflow qui marche

### Phase 6: Git + Documentation (Priorité 1)
- [ ] Commit chaque fix majeur
- [ ] Créer README_QUICK_START.md
- [ ] Lister commandes qui marchent
- [ ] Documenter workarounds

## Commits à Faire

1. `Fix: DataLoader retourne dict au lieu de tuple`
2. `Fix: Import dans training_cli.py`
3. `Fix: Implémentation get_complexity() DeepLabV3Plus`
4. `Feat: Script training minimal fonctionnel`
5. `Test: Validation workflow end-to-end`
6. `Docs: Guide quick start workflow`

## Notes
- Utiliser les outils existants autant que possible
- Ne pas créer 50000 fichiers inutiles
- Simple, efficace, propre
- Git commit réguliers
