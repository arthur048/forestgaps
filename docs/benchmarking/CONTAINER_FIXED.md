# ✅ Container forestgaps-main corrigé !

Le container `forestgaps-main` reste maintenant actif et accessible.

## Changement effectué

**Fichier modifié** : `docker/docker-compose.yml`

```yaml
forestgaps:
  # ...
  command: tail -f /dev/null  # ← Ligne ajoutée
```

Cette commande maintient le container actif indéfiniment sans consommer de ressources.

## Comment utiliser maintenant

### Option 1 : Depuis Windows PowerShell (RECOMMANDÉ)

```powershell
# Tester que tout fonctionne
.\test_docker.ps1

# Entrer dans le container
docker exec -it forestgaps-main bash

# Une fois dans le container (root@xxxxxx:/app#) :
python scripts/benchmark_quick_test.py --experiment-name "test"
```

### Option 2 : Exécuter directement (sans entrer)

```powershell
# Test rapide
docker exec -it forestgaps-main python scripts/benchmark_quick_test.py --experiment-name "test_$(Get-Date -Format 'yyyyMMdd')"

# Benchmark complet
docker exec -it forestgaps-main python scripts/benchmark_full.py --experiment-name "prod" --epochs 50
```

## Vérification

```powershell
# Container tourne ?
docker ps --filter "name=forestgaps-main"

# Logs
docker logs forestgaps-main

# Test import
docker exec forestgaps-main python -c "import forestgaps; print('OK')"
```

## Problèmes connus

### Git Bash transforme les chemins

**Symptôme** :
```
ls: cannot access 'C:/Users/.../Git/app/': No such file or directory
```

**Solution** : Utiliser PowerShell au lieu de Git Bash

```powershell
# PowerShell ✓
docker exec forestgaps-main ls /app/

# Git Bash ✗ (transforme /app/ en chemin Windows)
```

### Module forestgaps.environment non trouvé

**Vérification** :
```powershell
docker exec forestgaps-main python -c "import sys; import forestgaps; print(forestgaps.__path__)"
```

Si le package est en mode namespace, les sous-modules peuvent ne pas être importables directement.

**Solution temporaire** :
```powershell
# Réinstaller en mode editable
docker exec forestgaps-main pip install -e /app/
```

## Prochaine étape

Lance ton premier benchmark :

```powershell
docker exec -it forestgaps-main bash

# Dans le container :
python scripts/benchmark_quick_test.py --experiment-name "premier_test"
```

📊 Résultats dans : `outputs/benchmarks/<timestamp>_premier_test/`
