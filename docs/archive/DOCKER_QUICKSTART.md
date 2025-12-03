# 🐳 ForestGaps Docker - Guide de Démarrage Rapide

Guide ultra-simple pour démarrer avec Docker, même sans expérience Docker !

## ⚡ Démarrage en 3 Étapes

### 1️⃣ Build l'Image (10-15 min la première fois)

**Avec les scripts :**
```bash
./scripts/docker-build.sh
```

**Ou avec docker-compose :**
```bash
docker-compose -f docker/docker-compose.yml build
```

☕ Allez prendre un café, ça télécharge PyTorch + GDAL...

### 2️⃣ Vérifier que Tout Marche

```bash
./scripts/docker-test.sh
```

Vous devriez voir 7 tests passer ✅

### 3️⃣ C'est Prêt !

```bash
# Ouvrir un shell interactif
./scripts/docker-run.sh shell

# Ou lancer Jupyter
./scripts/docker-run.sh jupyter
```

---

## 🎯 Deux Façons d'Utiliser Docker

### Option A : Scripts Simplifiés (Recommandé)

**Avantage :** Super simple, pas besoin de connaître Docker

```bash
# Build l'image
./scripts/docker-build.sh

# Tester l'image
./scripts/docker-test.sh

# Ouvrir un shell
./scripts/docker-run.sh shell

# Lancer Jupyter (http://localhost:8888, token: forestgaps)
./scripts/docker-run.sh jupyter

# Entraîner un modèle
./scripts/docker-run.sh train --data-dir ./data --models-dir ./models

# Inférence
./scripts/docker-run.sh inference --data-dir ./data --models-dir ./models

# Tests
./scripts/docker-run.sh test
```

### Option B : Docker Compose (Traditionnel)

**Avantage :** Plus de contrôle, standard Docker

```bash
# Build et démarrer le container
docker-compose -f docker/docker-compose.yml up -d

# Voir les logs en direct
docker-compose -f docker/docker-compose.yml logs -f

# Ouvrir un shell dans le container
docker-compose -f docker/docker-compose.yml exec forestgaps /bin/bash

# Exécuter une commande
docker-compose -f docker/docker-compose.yml exec forestgaps python -m forestgaps.cli.train

# Arrêter le container
docker-compose -f docker/docker-compose.yml down
```

---

## 🖥️ Workflows Courants

### Développement Interactif

```bash
# Ouvrir Jupyter
./scripts/docker-run.sh jupyter

# Dans un autre terminal, voir les logs TensorBoard
docker run --rm -it \
  -p 6006:6006 \
  -v $(pwd)/logs:/app/logs:ro \
  forestgaps:latest \
  tensorboard --logdir=/app/logs --host=0.0.0.0
```

Accès :
- Jupyter : http://localhost:8888
- TensorBoard : http://localhost:6006

### Entraînement d'un Modèle

```bash
# Avec vos données dans ./data et ./models
./scripts/docker-run.sh train \
  --data-dir ./data \
  --models-dir ./models

# Les checkpoints sont sauvegardés dans ./models
# Les logs dans ./logs
```

### Inférence sur Nouvelles Données

```bash
./scripts/docker-run.sh inference \
  --data-dir ./nouvelles_donnees \
  --models-dir ./models \
  --outputs-dir ./predictions
```

Les prédictions seront dans `./predictions`

---

## 🔧 Customisation

### Utiliser Plus de CPU Cores

Éditez `docker/docker-compose.yml` :
```yaml
deploy:
  resources:
    limits:
      cpus: '16'  # Au lieu de 8
```

Puis :
```bash
docker-compose -f docker/docker-compose.yml up -d --force-recreate
```

### Modifier les Requirements

1. Éditez `requirements/requirements.txt`
2. Rebuild :
   ```bash
   ./scripts/docker-build.sh
   ```

### Live Code Editing (Sans Rebuild)

Décommentez dans `docker/docker-compose.yml` :
```yaml
volumes:
  - ../forestgaps:/app/forestgaps:rw  # <-- Décommenter
  - ../tests:/app/tests:rw            # <-- Décommenter
```

Relancez :
```bash
docker-compose -f docker/docker-compose.yml up -d
```

Vos modifications dans `forestgaps/` sont maintenant live !

---

## 🆘 Problèmes Fréquents

### "Cannot connect to Docker daemon"

**Solution :** Démarrez Docker Desktop

### "No GPU detected" (mais vous avez un GPU)

**Solutions :**
1. Vérifiez : `nvidia-smi` (doit fonctionner)
2. Installez nvidia-container-toolkit (voir [docker/README.md](docker/README.md#configuration-gpu))
3. Relancez Docker Desktop

### "No space left on device"

**Solutions :**
1. Libérez de l'espace disque
2. Nettoyez Docker :
   ```bash
   docker system prune -a
   ```

### Build très lent / échoue

**Solutions :**
1. Vérifiez votre connexion Internet (télécharge ~3 GB)
2. Rebuild sans cache :
   ```bash
   ./scripts/docker-build.sh --no-cache
   ```

---

## 📚 Documentation Complète

- **Guide détaillé :** [docker/README.md](docker/README.md)
- **Plan d'implémentation :** `.claude/plans/glowing-honking-gem.md`
- **Troubleshooting :** [docker/README.md#troubleshooting](docker/README.md#troubleshooting)

---

## 💡 Tips

1. **Première fois :** Le build prend 10-15 min, c'est normal !
2. **Runs suivants :** Quasi instantanés grâce au cache Docker
3. **GPU :** Auto-détecté, pas besoin de configuration manuelle
4. **Data :** Montez vos données en read-only (`ro`) pour sécurité
5. **Logs :** TensorBoard logs dans `./logs`, visualisez avec `tensorboard --logdir=./logs`

---

## ✅ Checklist Démarrage

- [ ] Docker Desktop installé et lancé
- [ ] (GPU) nvidia-smi fonctionne
- [ ] (GPU) nvidia-container-toolkit installé
- [ ] Build réussi : `./scripts/docker-build.sh`
- [ ] Tests passent : `./scripts/docker-test.sh`
- [ ] Shell fonctionne : `./scripts/docker-run.sh shell`

**Tout est vert ?** Vous êtes prêt ! 🚀

---

**Questions ?**
- Voir [docker/README.md](docker/README.md) pour plus de détails
- Ouvrir une issue sur GitHub
