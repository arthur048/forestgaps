# Interface CLI pour l'entraînement
"""
Interface en ligne de commande pour l'entraînement des modèles ForestGaps-DL.

Ce module fournit une interface en ligne de commande pour les fonctionnalités
d'entraînement des modèles du workflow ForestGaps-DL.
"""

import os
import argparse
import logging
import torch
from typing import Dict, List, Optional, Any

from config import ConfigManager
from environment import get_environment
from utils.errors import ErrorHandler
from utils.profiling.benchmarks import optimize_dataloader_params


def setup_parser() -> argparse.ArgumentParser:
    """
    Configure le parseur d'arguments pour l'interface CLI d'entraînement.
    
    Returns:
        argparse.ArgumentParser: Parseur d'arguments configuré.
    """
    parser = argparse.ArgumentParser(
        description="ForestGaps-DL - Entraînement des modèles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Arguments généraux
    parser.add_argument('--config', type=str, default='config/defaults/training.yaml',
                        help='Chemin vers le fichier de configuration')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Niveau de journalisation')
    parser.add_argument('--output-dir', type=str, help='Répertoire de sortie (remplace la valeur de la configuration)')
    
    # Sous-commandes
    subparsers = parser.add_subparsers(dest='command', help='Commande à exécuter')
    
    # Commande: train
    train_parser = subparsers.add_parser('train', help='Entraîner un modèle')
    train_parser.add_argument('--model-type', type=str, default='unet',
                             choices=['unet', 'unet_film', 'unet_cbam', 'unet_film_cbam', 'unet_all'],
                             help='Type de modèle à entraîner')
    train_parser.add_argument('--data-dir', type=str, required=True,
                             help='Répertoire contenant les données prétraitées')
    train_parser.add_argument('--epochs', type=int, help='Nombre d\'époques d\'entraînement')
    train_parser.add_argument('--batch-size', type=int, help='Taille des batchs')
    train_parser.add_argument('--learning-rate', type=float, help='Taux d\'apprentissage')
    train_parser.add_argument('--resume', type=str, help='Chemin vers un checkpoint pour reprendre l\'entraînement')
    train_parser.add_argument('--optimize-dataloader', action='store_true',
                             help='Optimiser les paramètres du DataLoader')
    
    # Commande: evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Évaluer un modèle')
    eval_parser.add_argument('--model-path', type=str, required=True,
                            help='Chemin vers le modèle à évaluer')
    eval_parser.add_argument('--data-dir', type=str, required=True,
                            help='Répertoire contenant les données de test')
    eval_parser.add_argument('--threshold', type=float, default=0.5,
                            help='Seuil de probabilité pour la segmentation')
    eval_parser.add_argument('--visualize', action='store_true',
                            help='Générer des visualisations')
    
    # Commande: export
    export_parser = subparsers.add_parser('export', help='Exporter un modèle')
    export_parser.add_argument('--model-path', type=str, required=True,
                              help='Chemin vers le modèle à exporter')
    export_parser.add_argument('--format', type=str, choices=['onnx', 'torchscript'],
                              default='onnx', help='Format d\'exportation')
    export_parser.add_argument('--input-shape', type=str, default='1,1,256,256',
                              help='Forme de l\'entrée du modèle (format: batch,channels,height,width)')
    
    # Commande: benchmark
    benchmark_parser = subparsers.add_parser('benchmark', help='Évaluer les performances d\'un modèle')
    benchmark_parser.add_argument('--model-path', type=str, required=True,
                                 help='Chemin vers le modèle à évaluer')
    benchmark_parser.add_argument('--batch-sizes', type=str, default='1,4,16,32,64',
                                 help='Tailles de batch à tester (séparées par des virgules)')
    benchmark_parser.add_argument('--repetitions', type=int, default=100,
                                 help='Nombre de répétitions pour chaque test')
    
    return parser


def run_train_command(args: argparse.Namespace, config: Dict, error_handler: ErrorHandler) -> None:
    """
    Exécute la commande d'entraînement d'un modèle.
    
    Args:
        args: Arguments de la ligne de commande.
        config: Configuration.
        error_handler: Gestionnaire d'erreurs.
    """
    try:
        # Mettre à jour la configuration avec les arguments de la ligne de commande
        if args.epochs is not None:
            config['training']['epochs'] = args.epochs
        if args.batch_size is not None:
            config['training']['batch_size'] = args.batch_size
        if args.learning_rate is not None:
            config['training']['learning_rate'] = args.learning_rate
        if args.model_type is not None:
            config['model']['type'] = args.model_type
        if args.output_dir is not None:
            config['output_dir'] = args.output_dir
        
        # Déterminer le répertoire de sortie
        output_dir = config.get('output_dir', 'output/models')
        os.makedirs(output_dir, exist_ok=True)
        
        # Charger les données
        logging.info(f"Chargement des données depuis {args.data_dir}")
        from data.loaders.factory import create_data_loaders
        
        # Optimiser les paramètres du DataLoader si demandé
        if args.optimize_dataloader:
            logging.info("Optimisation des paramètres du DataLoader...")
            from data.datasets.gap_dataset import GapDataset
            
            # Charger un petit échantillon du dataset pour l'optimisation
            sample_dataset = GapDataset(args.data_dir, config['training']['thresholds'])
            
            # Optimiser les paramètres
            optimal_params = optimize_dataloader_params(
                sample_dataset,
                config['training']['batch_size'],
                max_workers=config.get('training', {}).get('max_workers', 16)
            )
            
            # Mettre à jour la configuration
            config['training']['num_workers'] = optimal_params['num_workers']
            config['training']['prefetch_factor'] = optimal_params['prefetch_factor']
            
            logging.info(f"Paramètres optimaux: {optimal_params}")
        
        # Créer les DataLoaders
        train_loader, val_loader, test_loader = create_data_loaders(
            config,
            data_dir=args.data_dir
        )
        
        # Créer le modèle
        logging.info(f"Création du modèle de type {config['model']['type']}")
        from models.registry import ModelRegistry
        
        model = ModelRegistry.create(config['model']['type'], **config['model'])
        
        # Déplacer le modèle sur le périphérique approprié
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Créer la fonction de perte
        logging.info("Création de la fonction de perte")
        from training.loss.factory import create_loss_function
        
        criterion = create_loss_function(config['training']['loss'])
        
        # Créer l'optimiseur
        logging.info("Création de l'optimiseur")
        from training.optimization.factory import create_optimizer
        
        optimizer = create_optimizer(
            model.parameters(),
            config['training']['optimizer']
        )
        
        # Créer le scheduler
        logging.info("Création du scheduler")
        from training.optimization.factory import create_scheduler
        
        scheduler = create_scheduler(
            optimizer,
            config['training']['scheduler'],
            len(train_loader)
        )
        
        # Charger un checkpoint si demandé
        start_epoch = 0
        if args.resume:
            logging.info(f"Chargement du checkpoint {args.resume}")
            from utils.io.serialization import load_model
            
            checkpoint = load_model(args.resume, model, optimizer, device)
            start_epoch = checkpoint.get('epoch', 0) + 1
            logging.info(f"Reprise de l'entraînement à l'époque {start_epoch}")
        
        # Initialiser le système de monitoring
        logging.info("Initialisation du système de monitoring")
        from utils.visualization.tensorboard import MonitoringSystem
        
        monitoring = MonitoringSystem(
            log_dir=os.path.join(output_dir, 'logs'),
            config=config
        )
        
        # Enregistrer la configuration
        monitoring.log_config(config)
        
        # Enregistrer le graphe du modèle
        monitoring.log_model_graph(model)
        
        # Créer le trainer
        logging.info("Création du trainer")
        from training.trainer import Trainer
        
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            monitoring=monitoring
        )
        
        # Entraîner le modèle
        logging.info("Début de l'entraînement")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['training']['epochs'],
            start_epoch=start_epoch,
            checkpoint_dir=os.path.join(output_dir, 'checkpoints')
        )
        
        # Évaluer le modèle sur l'ensemble de test
        logging.info("Évaluation du modèle sur l'ensemble de test")
        test_metrics = trainer.evaluate(test_loader)
        
        # Afficher les métriques de test
        logging.info("Métriques de test:")
        for metric_name, value in test_metrics.items():
            if metric_name != 'threshold_metrics' and metric_name != 'confusion_data':
                logging.info(f"  {metric_name}: {value:.4f}")
        
        # Sauvegarder les métriques de test
        from utils.io.serialization import save_json
        metrics_path = os.path.join(output_dir, 'test_metrics.json')
        save_json(test_metrics, metrics_path)
        
        logging.info(f"Métriques de test sauvegardées dans {metrics_path}")
        logging.info("Entraînement terminé avec succès")
    except Exception as e:
        error_handler.handle(e, context={'command': 'train', 'args': vars(args)})
        logging.error(f"Erreur lors de l'entraînement du modèle: {str(e)}")


def run_evaluate_command(args: argparse.Namespace, config: Dict, error_handler: ErrorHandler) -> None:
    """
    Exécute la commande d'évaluation d'un modèle.
    
    Args:
        args: Arguments de la ligne de commande.
        config: Configuration.
        error_handler: Gestionnaire d'erreurs.
    """
    try:
        # Déterminer le répertoire de sortie
        output_dir = args.output_dir or config.get('output_dir', 'output/evaluation')
        os.makedirs(output_dir, exist_ok=True)
        
        # Charger le modèle
        logging.info(f"Chargement du modèle {args.model_path}")
        from utils.io.serialization import load_model
        from models.registry import ModelRegistry
        
        # Déterminer le périphérique
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Charger le checkpoint
        checkpoint = load_model(args.model_path, device=device)
        
        # Créer le modèle
        model_class = checkpoint.get('model_class', 'UNet')
        model = ModelRegistry.create(model_class)
        
        # Charger l'état du modèle
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Charger les données de test
        logging.info(f"Chargement des données de test depuis {args.data_dir}")
        from data.loaders.factory import create_test_loader
        
        test_loader = create_test_loader(
            config,
            data_dir=args.data_dir
        )
        
        # Évaluer le modèle
        logging.info("Évaluation du modèle")
        from training.metrics.segmentation import SegmentationMetrics
        
        metrics_calculator = SegmentationMetrics(device=device)
        
        # Parcourir les données de test
        with torch.no_grad():
            for dsm, threshold, target in test_loader:
                dsm = dsm.to(device)
                threshold = threshold.to(device)
                target = target.to(device)
                
                # Prédiction
                output = model(dsm, threshold)
                
                # Mettre à jour les métriques
                metrics_calculator.update(output, target)
                
                # Mettre à jour les métriques par seuil
                threshold_value = threshold[0].item() * max(test_loader.dataset.thresholds)
                closest_threshold = min(test_loader.dataset.thresholds, key=lambda x: abs(x - threshold_value))
                metrics_calculator.update_by_threshold(output, target, closest_threshold)
        
        # Calculer les métriques finales
        metrics = metrics_calculator.compute()
        
        # Afficher les métriques
        logging.info("Métriques d'évaluation:")
        for metric_name, value in metrics.items():
            if metric_name != 'threshold_metrics' and metric_name != 'confusion_data':
                logging.info(f"  {metric_name}: {value:.4f}")
        
        # Sauvegarder les métriques
        from utils.io.serialization import save_json
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
        save_json(metrics, metrics_path)
        
        logging.info(f"Métriques d'évaluation sauvegardées dans {metrics_path}")
        
        # Générer des visualisations si demandé
        if args.visualize:
            logging.info("Génération des visualisations")
            
            # Visualiser les métriques par seuil
            from utils.visualization.plots import visualize_metrics_by_threshold
            
            metrics_plot_path = os.path.join(output_dir, 'metrics_by_threshold.png')
            visualize_metrics_by_threshold(metrics, save_path=metrics_plot_path)
            
            # Visualiser la matrice de confusion
            from utils.visualization.plots import visualize_confusion_matrix
            
            confusion_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
            visualize_confusion_matrix(metrics['confusion_data'], save_path=confusion_plot_path)
            
            # Visualiser des exemples de prédictions
            from utils.visualization.maps import visualize_predictions_grid
            
            predictions_plot_path = os.path.join(output_dir, 'predictions_grid.png')
            visualize_predictions_grid(model, test_loader, device, config, num_samples=5)
            
            logging.info(f"Visualisations sauvegardées dans {output_dir}")
    except Exception as e:
        error_handler.handle(e, context={'command': 'evaluate', 'args': vars(args)})
        logging.error(f"Erreur lors de l'évaluation du modèle: {str(e)}")


def run_export_command(args: argparse.Namespace, config: Dict, error_handler: ErrorHandler) -> None:
    """
    Exécute la commande d'exportation d'un modèle.
    
    Args:
        args: Arguments de la ligne de commande.
        config: Configuration.
        error_handler: Gestionnaire d'erreurs.
    """
    try:
        # Déterminer le répertoire de sortie
        output_dir = args.output_dir or config.get('output_dir', 'output/exported_models')
        os.makedirs(output_dir, exist_ok=True)
        
        # Charger le modèle
        logging.info(f"Chargement du modèle {args.model_path}")
        from utils.io.serialization import load_model
        from models.registry import ModelRegistry
        
        # Déterminer le périphérique
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Charger le checkpoint
        checkpoint = load_model(args.model_path, device=device)
        
        # Créer le modèle
        model_class = checkpoint.get('model_class', 'UNet')
        model = ModelRegistry.create(model_class)
        
        # Charger l'état du modèle
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Analyser la forme de l'entrée
        input_shape = tuple(map(int, args.input_shape.split(',')))
        threshold_shape = (input_shape[0], 1)  # Forme du tenseur de seuil
        
        # Déterminer le chemin de sortie
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        
        if args.format == 'onnx':
            # Exporter au format ONNX
            logging.info(f"Exportation du modèle au format ONNX")
            from utils.io.serialization import export_model_to_onnx
            
            output_path = os.path.join(output_dir, f"{model_name}.onnx")
            export_model_to_onnx(model, output_path, input_shape, threshold_shape)
            
            logging.info(f"Modèle exporté au format ONNX: {output_path}")
        elif args.format == 'torchscript':
            # Exporter au format TorchScript
            logging.info(f"Exportation du modèle au format TorchScript")
            from utils.io.serialization import export_model_to_torchscript
            
            output_path = os.path.join(output_dir, f"{model_name}.pt")
            export_model_to_torchscript(model, output_path, input_shape, threshold_shape)
            
            logging.info(f"Modèle exporté au format TorchScript: {output_path}")
    except Exception as e:
        error_handler.handle(e, context={'command': 'export', 'args': vars(args)})
        logging.error(f"Erreur lors de l'exportation du modèle: {str(e)}")


def run_benchmark_command(args: argparse.Namespace, config: Dict, error_handler: ErrorHandler) -> None:
    """
    Exécute la commande de benchmark d'un modèle.
    
    Args:
        args: Arguments de la ligne de commande.
        config: Configuration.
        error_handler: Gestionnaire d'erreurs.
    """
    try:
        # Déterminer le répertoire de sortie
        output_dir = args.output_dir or config.get('output_dir', 'output/benchmarks')
        os.makedirs(output_dir, exist_ok=True)
        
        # Charger le modèle
        logging.info(f"Chargement du modèle {args.model_path}")
        from utils.io.serialization import load_model
        from models.registry import ModelRegistry
        
        # Déterminer le périphérique
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Charger le checkpoint
        checkpoint = load_model(args.model_path, device=device)
        
        # Créer le modèle
        model_class = checkpoint.get('model_class', 'UNet')
        model = ModelRegistry.create(model_class)
        
        # Charger l'état du modèle
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Analyser les tailles de batch
        batch_sizes = list(map(int, args.batch_sizes.split(',')))
        
        # Créer un tenseur d'entrée d'exemple
        sample_input = torch.randn(1, 1, 256, 256, device=device)
        
        # Exécuter le benchmark
        logging.info(f"Exécution du benchmark avec les tailles de batch: {batch_sizes}")
        from utils.profiling.benchmarks import benchmark_model_architectures
        
        results = benchmark_model_architectures(
            {model_class: lambda: model},
            sample_input,
            batch_sizes=batch_sizes,
            repetitions=args.repetitions
        )
        
        # Sauvegarder les résultats
        from utils.io.serialization import save_json
        results_path = os.path.join(output_dir, f"{model_class}_benchmark.json")
        save_json(results, results_path)
        
        logging.info(f"Résultats du benchmark sauvegardés dans {results_path}")
    except Exception as e:
        error_handler.handle(e, context={'command': 'benchmark', 'args': vars(args)})
        logging.error(f"Erreur lors du benchmark du modèle: {str(e)}")


def main() -> None:
    """Point d'entrée principal pour l'interface CLI d'entraînement."""
    # Configurer le parseur d'arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # Configurer la journalisation
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Charger la configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Initialiser le gestionnaire d'erreurs
    error_handler = ErrorHandler(
        log_file=config.get('error_log', 'logs/training_errors.log'),
        verbose=True
    )
    
    # Initialiser l'environnement
    env = get_environment()
    env.setup()
    
    # Exécuter la commande appropriée
    if args.command == 'train':
        run_train_command(args, config, error_handler)
    elif args.command == 'evaluate':
        run_evaluate_command(args, config, error_handler)
    elif args.command == 'export':
        run_export_command(args, config, error_handler)
    elif args.command == 'benchmark':
        run_benchmark_command(args, config, error_handler)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
