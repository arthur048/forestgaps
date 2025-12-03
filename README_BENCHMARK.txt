================================================================================
                    FORESTGAPS - BENCHMARKING SETUP
================================================================================

✅ INFRASTRUCTURE COMPLÈTE ET OPÉRATIONNELLE

📁 STRUCTURE
------------
forestgaps-dl/
├── 📊 data/                       Données d'entraînement (16 plots)
│   └── data_external_test/        Données externes (SODEFOR)
├── 📈 logs/                       Logs TensorBoard
├── 📦 outputs/                    Résultats des expériences
├── 🤖 models/                     Modèles sauvegardés
├── 🐳 docker/                     Configuration Docker
└── 🔧 scripts/                    Scripts de benchmarking

🚀 COMMANDES ESSENTIELLES
--------------------------

1️⃣ LANCER TENSORBOARD
   cd docker/
   docker-compose up -d tensorboard
   → http://localhost:6006

2️⃣ TEST RAPIDE (5-10 min)
   docker-compose run --rm forestgaps python scripts/benchmark_quick_test.py \
     --experiment-name "test"

3️⃣ BENCHMARK COMPLET (4-8h)
   docker-compose run --rm forestgaps python scripts/benchmark_full.py \
     --experiment-name "comparison_complete"

📚 DOCUMENTATION
----------------
1. QUICK_START_BENCHMARK.md    Guide de démarrage
2. BENCHMARKING_GUIDE.md       Organisation complète
3. SETUP_COMPLETE.md           Ce setup
4. scripts/README.md           Documentation scripts

🎯 WORKFLOW RECOMMANDÉ
----------------------
Phase 1: Test rapide (AUJOURD'HUI)
  └─> Valider que tout fonctionne (5-10 min)

Phase 2: Benchmark complet (DEMAIN)
  └─> Comparer tous les modèles (4-8h)

Phase 3: Évaluation externe
  └─> Tester sur données SODEFOR

Phase 4: Production
  └─> Sauvegarder le meilleur modèle

📊 MODÈLES DISPONIBLES
-----------------------
✓ U-Net Base
✓ U-Net FiLM  
✓ DeepLabV3+ Base
✓ DeepLabV3+ Threshold

🎓 MÉTRIQUES SUIVIES
--------------------
• IoU (Intersection over Union)
• F1-Score
• Precision / Recall
• Training time
• Convergence speed

🔗 LIENS IMPORTANTS
-------------------
TensorBoard: http://localhost:6006
Jupyter Lab: http://localhost:8888

🆘 AIDE RAPIDE
--------------
Problème CUDA       → --batch-size 2
TensorBoard vide    → Attendre 1-2 min
Container crash     → docker-compose logs forestgaps
Pas de GPU          → nvidia-smi

================================================================================
           TU ES PRÊT ! LANCE TON PREMIER BENCHMARK MAINTENANT 🚀
================================================================================

COMMANDE POUR COMMENCER:
  cd docker/ && docker-compose up -d tensorboard && \
  docker-compose run --rm forestgaps python scripts/benchmark_quick_test.py \
    --experiment-name "test_initial"

================================================================================
