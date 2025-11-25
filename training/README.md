# Module d'entraînement (training)

Ce module fournit des classes et des fonctions pour l'entraînement des modèles de segmentation dans le cadre du projet forestgaps. Il implémente un système modulaire et extensible pour entrainer, évaluer et tester des modèles de détection de trouées forestières.

## Structure du module

```
training/
├── __init__.py               # Point d'entrée unifié
├── trainer.py                # Classe principale d'entraînement
├── metrics/                  # Métriques et évaluation
│   ├── __init__.py
│   ├── segmentation.py       # Métriques de segmentation
│   └── classification.py     # Métriques par seuil
├── loss/                     # Fonctions de perte
│   ├── __init__.py
│   ├── combined.py           # Pertes combinées (Focal+Dice)
│   └── factory.py            # Création de fonctions de perte
├── callbacks/                # Système de callbacks
│   ├── __init__.py
│   ├── base.py               # Classe de base des callbacks
│   ├── logging.py            # Callbacks de journalisation
│   ├── checkpointing.py      # Sauvegarde des points de contrôle
│   └── visualization.py      # Visualisation pendant l'entraînement
└── optimization/             # Optimisation de l'entraînement
    ├── __init__.py
    ├── lr_schedulers.py      # Schedulers de learning rate
    └── regularization.py     # Techniques de régularisation
```

## Fonctionnalités principales

### Classe Trainer

La classe `Trainer` est le point central du module, encapsulant toute la logique d'entraînement, de validation et de test. Elle s'occupe également de la gestion des points de contrôle, des métriques et de l'optimisation.

```python
from forestgaps.training import Trainer

trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)

history = trainer.train()
test_metrics = trainer.test()
```

### Système de métriques

Le module inclut un système avancé de métriques pour évaluer les performances des modèles de segmentation, avec un support spécifique pour les métriques par seuil de hauteur.

```python
from forestgaps.training.metrics.segmentation import SegmentationMetrics
from forestgaps.training.metrics.classification import ThresholdMetrics

# Métriques générales de segmentation
metrics = SegmentationMetrics(device="cuda")
metrics.update(predictions, targets)
results = metrics.compute()

# Métriques par seuil de hauteur
threshold_metrics = ThresholdMetrics(thresholds=[5, 10, 15, 20])
threshold_metrics.update(predictions, targets, 0.5, threshold_value=10)
threshold_results = threshold_metrics.compute()
```

### Fonctions de perte optimisées

Le module propose plusieurs fonctions de perte adaptées à la segmentation, notamment une perte combinée Focal+Dice avec pondération par seuil.

```python
from forestgaps.training.loss.combined import CombinedFocalDiceLoss
from forestgaps.training.loss.factory import create_loss_function

# Utilisation directe
loss_fn = CombinedFocalDiceLoss(alpha=0.5, gamma=2.0)

# Création à partir de la configuration
loss_fn = create_loss_function(config)
```

### Système extensible de callbacks

Un système de callbacks permet de personnaliser le comportement de l'entraînement, avec des callbacks prédéfinis pour le logging, les points de contrôle, la visualisation, etc.

```python
from forestgaps.training.callbacks.logging import LoggingCallback
from forestgaps.training.callbacks.checkpointing import CheckpointingCallback

callbacks = [
    LoggingCallback(log_dir="logs"),
    CheckpointingCallback(checkpoint_dir="checkpoints")
]

trainer = Trainer(..., callbacks=callbacks)
```

### Techniques d'optimisation avancées

Le module inclut des techniques d'optimisation avancées comme le scheduler de learning rate adaptatif, le gradient clipping ou la régularisation composite.

```python
from forestgaps.training.optimization.regularization import CompositeRegularization
from forestgaps.training.optimization.lr_schedulers import create_scheduler

# Régularisation avancée
reg = CompositeRegularization(
    model=model,
    dropout_rate=0.2,
    weight_decay=1e-4,
    drop_path_rate=0.1
)

# Création d'un scheduler adapté
scheduler = create_scheduler(optimizer, config, len(train_loader))
```

## Utilisation

### Entraînement basique

```python
from forestgaps.training import train_model

model, history = train_model(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)
```

### Contrôle plus fin avec la classe Trainer

```python
from forestgaps.training import Trainer

# Initialisation
trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader
)

# Entraînement
history = trainer.train()

# Sauvegarde d'un point de contrôle
trainer.save_checkpoint("checkpoints/model_best.pt")

# Chargement d'un point de contrôle
trainer.load_checkpoint("checkpoints/model_best.pt")

# Test
test_metrics = trainer.test()

# Prédiction
predictions = trainer.predict(inputs, thresholds)
```

## Techniques d'optimisation

Le module implémente plusieurs techniques d'optimisation avancées :

1. **Normalisation adaptative** : Bascule automatiquement entre BatchNorm et GroupNorm selon la taille du batch.
2. **Gradient Clipping** : Limite la norme des gradients pour stabiliser l'entraînement.
3. **DropPath (Stochastic Depth)** : Désactive aléatoirement des blocs entiers pendant l'entraînement.
4. **Schedulers de learning rate avancés** : Multiples stratégies comme le warmup, cosine annealing, etc.
5. **Régularisation composite** : Combinaison optimale de différentes techniques de régularisation.

## Exemples

### Configuration avancée

```python
# Configuration avec techniques avancées
config = Config()
config.epochs = 100
config.learning_rate = 1e-3
config.optimizer = 'adamw'
config.scheduler_type = 'cosine_warm'
config.warmup_epochs = 5
config.use_gradient_clipping = True
config.clip_value = 1.0
config.use_threshold_weights = True
config.use_droppath = True
config.droppath_final_prob = 0.2

# Entraînement avec cette configuration
trainer = Trainer(model=model, config=config, ...)
```

### Callbacks personnalisés

```python
# Callbacks personnalisés
class MyCustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Logique personnalisée à la fin de chaque époque
        print(f"Epoch {epoch} completed with metrics: {logs['val_metrics']}")

# Utilisation
callbacks = [MyCustomCallback()]
trainer = Trainer(..., callbacks=callbacks)
``` 