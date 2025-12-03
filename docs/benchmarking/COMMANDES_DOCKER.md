# Commandes Docker - ForestGaps Benchmarking

Guide pratique pour lancer les benchmarks dans Docker.

## 🐳 Prérequis

**Démarrer les services Docker :**
```bash
cd "g:\Mon Drive\forestgaps-dl\docker"
docker-compose up -d
```

Cela lance :
- ✅ Container principal `forestgaps-main`
- ✅ TensorBoard sur http://localhost:6006
- ✅ Jupyter Lab sur http://localhost:8888

## 📍 Tu es où ?

### Si tu es dans ton terminal Windows (PowerShell/Git Bash)
```bash
# Tu verras quelque chose comme :
PS G:\Mon Drive\forestgaps-dl>
# ou
user@machine MINGW64 /g/Mon Drive/forestgaps-dl
```

### Si tu es DANS le container Docker
```bash
# Tu verras quelque chose comme :
root@f8a4ada3a838:/app#
```

## 🚀 Lancer un benchmark

### **Option A : Depuis Windows (SANS entrer dans le container)**

#### Test rapide :
```bash
cd "g:\Mon Drive\forestgaps-dl"
docker exec -it forestgaps-main python /app/run_benchmark_quick.py \
  --experiment-name "test_windows"
```

#### Benchmark complet :
```bash
cd "g:\Mon Drive\forestgaps-dl"
docker exec -it forestgaps-main python /app/run_benchmark_full.py \
  --experiment-name "benchmark_complet" \
  --epochs 50
```

---

### **Option B : Depuis DANS le container (après avoir fait `docker exec -it forestgaps-main bash`)**

#### Entrer dans le container :
```bash
docker exec -it forestgaps-main bash
```

Tu es maintenant dans le container, tu verras :
```
root@f8a4ada3a838:/app#
```

#### Test rapide (DANS le container) :
```bash
python run_benchmark_quick.py --experiment-name "test_interne"
```

#### Benchmark complet (DANS le container) :
```bash
python run_benchmark_full.py \
  --experiment-name "benchmark_complet" \
  --epochs 50 \
  --batch-size 8
```

#### Sortir du container :
```bash
exit
```

---

## 📊 Voir les résultats

### **Depuis Windows (ton explorateur de fichiers)**
```
G:\Mon Drive\forestgaps-dl\outputs\benchmarks\
```

Ouvre le fichier :
```
outputs\benchmarks\<timestamp>_<nom>\reports\benchmark_report.html
```

### **Liste des benchmarks**
```bash
# Depuis Windows
dir "g:\Mon Drive\forestgaps-dl\outputs\benchmarks" /O-D

# Depuis le container
ls -lhtr /app/outputs/benchmarks/
```

---

## 🔍 Vérifications utiles

### Vérifier que les containers tournent
```bash
docker-compose ps
```

Tu dois voir :
```
forestgaps-main          running
forestgaps-tensorboard   running
forestgaps-jupyter       running
```

### Vérifier les logs en temps réel
```bash
docker-compose logs -f forestgaps
```

### Vérifier le GPU
```bash
docker exec forestgaps-main nvidia-smi
```

### Vérifier les données montées
```bash
docker exec forestgaps-main ls -lh /app/data/*.tif | head -5
```

---

## 🛠️ Paramètres disponibles

### Test rapide (`run_benchmark_quick.py`)
```bash
python run_benchmark_quick.py \
  --experiment-name "mon_test" \        # Nom de l'expérience
  --epochs 5 \                          # Nombre d'époques (défaut: 5)
  --batch-size 4 \                      # Taille batch (défaut: 4)
  --max-train-tiles 20 \                # Nb tuiles train (défaut: 20)
  --models "unet,unet_film" \           # Modèles à comparer
  --thresholds "5.0,10.0"               # Seuils de hauteur
```

### Benchmark complet (`run_benchmark_full.py`)
```bash
python run_benchmark_full.py \
  --experiment-name "benchmark_prod" \  # Nom (REQUIS)
  --epochs 50 \                         # Époques (défaut: 50)
  --batch-size 8 \                      # Batch size (défaut: 8)
  --models "unet,unet_film,deeplabv3_plus,deeplabv3_plus_threshold" \
  --thresholds "2.0,5.0,10.0,15.0"
```

---

## 💡 Exemples pratiques

### 1. Test ultra-rapide (2 minutes)
```bash
docker exec -it forestgaps-main python /app/run_benchmark_quick.py \
  --experiment-name "test_2min" \
  --epochs 2 \
  --max-train-tiles 10 \
  --models "unet"
```

### 2. Test standard (5-10 minutes)
```bash
docker exec -it forestgaps-main python /app/run_benchmark_quick.py \
  --experiment-name "test_standard"
```

### 3. Comparaison 2 modèles (30 minutes)
```bash
docker exec -it forestgaps-main python /app/run_benchmark_quick.py \
  --experiment-name "unet_vs_film" \
  --epochs 10 \
  --models "unet,unet_film"
```

### 4. Benchmark complet production (4-8 heures)
```bash
docker exec -it forestgaps-main python /app/run_benchmark_full.py \
  --experiment-name "production_$(date +%Y%m%d)" \
  --epochs 50
```

---

## 🐛 Résolution de problèmes

### Erreur "No module named 'forestgaps'"
```bash
# Entrer dans le container et installer
docker exec -it forestgaps-main bash
pip install -e .
exit
```

### Erreur "CUDA out of memory"
```bash
# Réduire le batch size
docker exec -it forestgaps-main python /app/run_benchmark_quick.py \
  --experiment-name "test" \
  --batch-size 2
```

### Container ne démarre pas
```bash
# Voir les logs
docker-compose logs forestgaps

# Redémarrer
docker-compose restart forestgaps
```

### TensorBoard ne s'affiche pas
```bash
# Redémarrer TensorBoard
docker-compose restart tensorboard

# Vérifier qu'il tourne
docker-compose ps tensorboard
```

---

## 📝 Workflow simple

**1. Démarrer Docker + TensorBoard**
```bash
cd "g:\Mon Drive\forestgaps-dl\docker"
docker-compose up -d
```

**2. Lancer un test rapide**
```bash
docker exec -it forestgaps-main python /app/run_benchmark_quick.py \
  --experiment-name "test_$(date +%Y%m%d_%H%M)"
```

**3. Surveiller sur TensorBoard**
```
→ http://localhost:6006
```

**4. Voir les résultats**
```bash
explorer.exe "G:\Mon Drive\forestgaps-dl\outputs\benchmarks"
```

**5. Lancer le benchmark complet si satisfait**
```bash
docker exec -it forestgaps-main python /app/run_benchmark_full.py \
  --experiment-name "production_v1" \
  --epochs 50
```

---

## 🎯 Commande ultime (tout en un)

```bash
cd "g:\Mon Drive\forestgaps-dl\docker" && \
docker-compose up -d && \
echo "✅ Docker lancé" && \
sleep 5 && \
docker exec -it forestgaps-main python /app/run_benchmark_quick.py \
  --experiment-name "test_rapide" && \
echo "✅ Benchmark terminé ! Résultats dans outputs/benchmarks/"
```

---

## 🔗 Liens utiles

- TensorBoard : http://localhost:6006
- Jupyter Lab : http://localhost:8888
- Outputs : `G:\Mon Drive\forestgaps-dl\outputs\benchmarks\`
- Logs : `G:\Mon Drive\forestgaps-dl\logs\benchmarks\`

---

## ⚡ Commandes courtes (alias)

Tu peux copier ces commandes dans un fichier `aliases.sh` :

```bash
# Test rapide
alias bench-test='docker exec -it forestgaps-main python /app/run_benchmark_quick.py'

# Benchmark complet
alias bench-full='docker exec -it forestgaps-main python /app/run_benchmark_full.py'

# Voir résultats
alias bench-results='ls -lhtr /g/Mon\ Drive/forestgaps-dl/outputs/benchmarks/'

# Entrer dans container
alias bench-shell='docker exec -it forestgaps-main bash'

# Logs
alias bench-logs='docker-compose logs -f forestgaps'
```

Puis utiliser :
```bash
bench-test --experiment-name "mon_test"
```

---

Voilà ! Tu as toutes les commandes pour lancer tes benchmarks facilement ! 🚀
