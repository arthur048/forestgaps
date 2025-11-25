"""
Module de callbacks de visualisation pour l'entraînement.

Ce module fournit des callbacks pour la visualisation des données, prédictions et métriques
pendant l'entraînement des modèles.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from .base import Callback


class VisualizationCallback(Callback):
    """
    Callback pour la visualisation des prédictions et métriques.
    
    Ce callback permet de générer des visualisations des prédictions du modèle
    et de l'évolution des métriques pendant l'entraînement.
    """
    
    def __init__(self, log_dir: str, val_dataset=None, 
                 num_samples: int = 4, save_frequency: int = 5,
                 use_tensorboard: bool = True, figure_size: Tuple[int, int] = (12, 10),
                 thresholds: Optional[List[float]] = None):
        """
        Initialise le callback de visualisation.
        
        Args:
            log_dir: Répertoire où sauvegarder les visualisations.
            val_dataset: Dataset de validation pour les visualisations.
            num_samples: Nombre d'échantillons à visualiser.
            save_frequency: Fréquence de sauvegarde des visualisations (en époques).
            use_tensorboard: Utiliser TensorBoard pour les visualisations.
            figure_size: Taille des figures matplotlib.
            thresholds: Liste des seuils de hauteur à visualiser.
        """
        super(VisualizationCallback, self).__init__()
        
        # Paramètres de base
        self.log_dir = log_dir
        self.val_dataset = val_dataset
        self.num_samples = min(num_samples, len(val_dataset) if val_dataset else num_samples)
        self.save_frequency = save_frequency
        self.use_tensorboard = use_tensorboard
        self.figure_size = figure_size
        self.thresholds = thresholds or [5, 10, 15, 20]
        
        # Créer les répertoires
        self.vis_dir = os.path.join(log_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Initialiser TensorBoard writer si nécessaire
        self.writer = None
        if use_tensorboard:
            self.writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))
        
        # Sélectionner des échantillons aléatoires pour la visualisation
        self.sample_indices = []
        if val_dataset:
            self.sample_indices = np.random.choice(
                len(val_dataset), min(self.num_samples, len(val_dataset)), replace=False
            ).tolist()
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé au début de l'entraînement.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        # Visualiser quelques exemples du dataset de validation
        if self.val_dataset and self.sample_indices:
            self._visualize_dataset_samples()
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de chaque époque.
        
        Args:
            epoch: Numéro de l'époque actuelle.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        # Vérifier si on doit générer des visualisations à cette époque
        if (epoch + 1) % self.save_frequency != 0:
            return
        
        if logs is None:
            return
        
        # Récupérer le modèle
        model = logs.get('model')
        device = logs.get('device')
        
        if model is None:
            return
        
        # Générer des visualisations de prédictions
        if self.val_dataset and self.sample_indices:
            self._visualize_predictions(model, device, epoch)
        
        # Visualiser les métriques
        train_metrics = logs.get('train_metrics', {})
        val_metrics = logs.get('val_metrics', {})
        
        if train_metrics and val_metrics:
            self._visualize_metrics(train_metrics, val_metrics, epoch)
        
        # Visualiser les métriques par seuil si disponibles
        if 'threshold_metrics' in val_metrics:
            self._visualize_threshold_metrics(val_metrics['threshold_metrics'], epoch)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de l'entraînement.
        
        Args:
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        # Générer un résumé des métriques si disponibles
        if logs and 'history' in logs:
            history = logs['history']
            self._visualize_history(history)
        
        # Fermer le writer TensorBoard
        if self.writer:
            self.writer.close()
    
    def _visualize_dataset_samples(self) -> None:
        """
        Visualise quelques exemples du dataset de validation.
        """
        fig, axes = plt.subplots(self.num_samples, 3, figsize=self.figure_size)
        
        for i, idx in enumerate(self.sample_indices):
            # Récupérer l'échantillon
            sample = self.val_dataset[idx]
            dsm = sample['dsm'] if isinstance(sample, dict) else sample[0]
            mask = sample['mask'] if isinstance(sample, dict) else sample[1]
            threshold = sample.get('threshold', 10.0) if isinstance(sample, dict) else 10.0
            
            # Convertir en numpy
            dsm_np = dsm.numpy() if isinstance(dsm, torch.Tensor) else dsm
            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
            
            # Ajuster les dimensions si nécessaire
            if len(dsm_np.shape) == 3:
                dsm_np = dsm_np[0]  # Prendre le premier canal
            if len(mask_np.shape) == 3:
                mask_np = mask_np[0]
            
            # Afficher les images
            if self.num_samples == 1:
                axes[0].imshow(dsm_np, cmap='terrain')
                axes[0].set_title(f"DSM (seuil: {threshold}m)")
                axes[0].axis('off')
                
                axes[1].imshow(mask_np, cmap='gray')
                axes[1].set_title("Masque")
                axes[1].axis('off')
                
                axes[2].set_visible(False)
            else:
                axes[i, 0].imshow(dsm_np, cmap='terrain')
                axes[i, 0].set_title(f"DSM (seuil: {threshold}m)")
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(mask_np, cmap='gray')
                axes[i, 1].set_title("Masque")
                axes[i, 1].axis('off')
                
                axes[i, 2].set_visible(False)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder la figure
        samples_path = os.path.join(self.vis_dir, 'dataset_samples.png')
        plt.savefig(samples_path)
        plt.close(fig)
        
        # Ajouter à TensorBoard
        if self.writer:
            self.writer.add_figure('Dataset/Samples', fig, 0)
    
    def _visualize_predictions(self, model, device, epoch: int) -> None:
        """
        Visualise les prédictions du modèle sur quelques échantillons.
        
        Args:
            model: Modèle à utiliser pour les prédictions.
            device: Dispositif sur lequel effectuer les prédictions.
            epoch: Numéro de l'époque actuelle.
        """
        # Mettre le modèle en mode évaluation
        model.eval()
        
        # Créer la figure
        fig, axes = plt.subplots(self.num_samples, 3, figsize=self.figure_size)
        
        with torch.no_grad():
            for i, idx in enumerate(self.sample_indices):
                # Récupérer l'échantillon
                sample = self.val_dataset[idx]
                dsm = sample['dsm'] if isinstance(sample, dict) else sample[0]
                mask = sample['mask'] if isinstance(sample, dict) else sample[1]
                threshold = sample.get('threshold', 10.0) if isinstance(sample, dict) else 10.0
                
                # Ajouter les dimensions batch et déplacer sur le device
                dsm_tensor = dsm.unsqueeze(0).to(device) if isinstance(dsm, torch.Tensor) else torch.tensor(dsm, device=device).unsqueeze(0)
                threshold_tensor = torch.tensor([threshold], device=device) if isinstance(threshold, (int, float)) else threshold.to(device)
                
                # Prédire
                pred = model(dsm_tensor, threshold_tensor)
                
                # Binariser les prédictions
                pred_bin = (pred > 0.5).float()
                
                # Convertir en numpy
                dsm_np = dsm.cpu().numpy() if isinstance(dsm, torch.Tensor) else dsm
                mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                pred_np = pred_bin.cpu().numpy()[0, 0] if pred_bin.shape[1] == 1 else pred_bin.cpu().numpy()[0]
                
                # Ajuster les dimensions si nécessaire
                if len(dsm_np.shape) == 3:
                    dsm_np = dsm_np[0]
                if len(mask_np.shape) == 3:
                    mask_np = mask_np[0]
                
                # Afficher les images
                if self.num_samples == 1:
                    axes[0].imshow(dsm_np, cmap='terrain')
                    axes[0].set_title(f"DSM (seuil: {threshold}m)")
                    axes[0].axis('off')
                    
                    axes[1].imshow(mask_np, cmap='gray')
                    axes[1].set_title("Référence")
                    axes[1].axis('off')
                    
                    axes[2].imshow(pred_np, cmap='gray')
                    axes[2].set_title("Prédiction")
                    axes[2].axis('off')
                else:
                    axes[i, 0].imshow(dsm_np, cmap='terrain')
                    axes[i, 0].set_title(f"DSM (seuil: {threshold}m)")
                    axes[i, 0].axis('off')
                    
                    axes[i, 1].imshow(mask_np, cmap='gray')
                    axes[i, 1].set_title("Référence")
                    axes[i, 1].axis('off')
                    
                    axes[i, 2].imshow(pred_np, cmap='gray')
                    axes[i, 2].set_title("Prédiction")
                    axes[i, 2].axis('off')
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder la figure
        pred_path = os.path.join(self.vis_dir, f'predictions_epoch_{epoch+1:03d}.png')
        plt.savefig(pred_path)
        plt.close(fig)
        
        # Ajouter à TensorBoard
        if self.writer:
            self.writer.add_figure(f'Predictions/Epoch_{epoch+1}', fig, epoch)
    
    def _visualize_metrics(self, train_metrics: Dict[str, float], 
                          val_metrics: Dict[str, float], epoch: int) -> None:
        """
        Visualise les métriques d'entraînement et de validation.
        
        Args:
            train_metrics: Métriques d'entraînement.
            val_metrics: Métriques de validation.
            epoch: Numéro de l'époque actuelle.
        """
        # Créer la figure
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        
        # Métriques principales
        main_metrics = ['loss', 'iou', 'f1', 'precision', 'recall']
        metrics_titles = ['Loss', 'IoU', 'F1-Score', 'Precision/Recall']
        
        # Première sous-figure: Loss
        if 'loss' in train_metrics and 'loss' in val_metrics:
            axes[0, 0].bar(['Train', 'Validation'], [train_metrics['loss'], val_metrics['loss']])
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_ylim(0, max(train_metrics['loss'], val_metrics['loss']) * 1.2)
        
        # Deuxième sous-figure: IoU
        if 'iou' in train_metrics and 'iou' in val_metrics:
            axes[0, 1].bar(['Train', 'Validation'], [train_metrics['iou'], val_metrics['iou']])
            axes[0, 1].set_title('IoU')
            axes[0, 1].set_ylim(0, 1)
        
        # Troisième sous-figure: F1-Score
        if 'f1' in train_metrics and 'f1' in val_metrics:
            axes[1, 0].bar(['Train', 'Validation'], [train_metrics['f1'], val_metrics['f1']])
            axes[1, 0].set_title('F1-Score')
            axes[1, 0].set_ylim(0, 1)
        
        # Quatrième sous-figure: Precision/Recall
        if all(m in train_metrics and m in val_metrics for m in ['precision', 'recall']):
            axes[1, 1].bar(['Train P', 'Val P', 'Train R', 'Val R'], 
                         [train_metrics['precision'], val_metrics['precision'], 
                          train_metrics['recall'], val_metrics['recall']])
            axes[1, 1].set_title('Precision/Recall')
            axes[1, 1].set_ylim(0, 1)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder la figure
        metrics_path = os.path.join(self.vis_dir, f'metrics_epoch_{epoch+1:03d}.png')
        plt.savefig(metrics_path)
        plt.close(fig)
        
        # Ajouter à TensorBoard
        if self.writer:
            self.writer.add_figure(f'Metrics/Epoch_{epoch+1}', fig, epoch)
    
    def _visualize_threshold_metrics(self, threshold_metrics: Dict[float, Dict[str, float]], 
                                   epoch: int) -> None:
        """
        Visualise les métriques par seuil de hauteur.
        
        Args:
            threshold_metrics: Métriques par seuil.
            epoch: Numéro de l'époque actuelle.
        """
        # Extraire les seuils et métriques
        thresholds = sorted(threshold_metrics.keys())
        metrics_to_plot = ['iou', 'f1', 'precision', 'recall']
        
        # Créer la figure
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        flat_axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            values = [threshold_metrics[t].get(metric, 0) for t in thresholds]
            flat_axes[i].plot(thresholds, values, 'o-')
            flat_axes[i].set_title(f'{metric.capitalize()} par seuil')
            flat_axes[i].set_xlabel('Seuil de hauteur (m)')
            flat_axes[i].set_ylabel(metric.capitalize())
            flat_axes[i].set_ylim(0, 1)
            flat_axes[i].grid(True)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder la figure
        thresh_path = os.path.join(self.vis_dir, f'threshold_metrics_epoch_{epoch+1:03d}.png')
        plt.savefig(thresh_path)
        plt.close(fig)
        
        # Ajouter à TensorBoard
        if self.writer:
            self.writer.add_figure(f'ThresholdMetrics/Epoch_{epoch+1}', fig, epoch)
    
    def _visualize_history(self, history: Dict[str, List]) -> None:
        """
        Visualise l'historique d'entraînement.
        
        Args:
            history: Historique d'entraînement.
        """
        # Vérifier que l'historique n'est pas vide
        if not history or 'train_loss' not in history or not history['train_loss']:
            return
        
        # Créer la figure pour les pertes
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], 'r-', label='Validation')
        ax1.set_title('Évolution de la perte')
        ax1.set_xlabel('Époques')
        ax1.set_ylabel('Perte')
        ax1.legend()
        ax1.grid(True)
        
        # Sauvegarder la figure
        loss_path = os.path.join(self.vis_dir, 'loss_history.png')
        plt.savefig(loss_path)
        plt.close(fig1)
        
        # Créer la figure pour les métriques
        fig2, axes2 = plt.subplots(2, 2, figsize=self.figure_size)
        flat_axes = axes2.flatten()
        
        metrics_to_plot = ['iou', 'f1', 'precision', 'recall']
        
        for i, metric in enumerate(metrics_to_plot):
            # Métriques d'entraînement
            if 'train_metrics' in history and history['train_metrics']:
                train_values = [m.get(metric, 0) for m in history['train_metrics']]
                flat_axes[i].plot(epochs, train_values, 'b-', label='Train')
            
            # Métriques de validation
            if 'val_metrics' in history and history['val_metrics']:
                val_values = [m.get(metric, 0) for m in history['val_metrics']]
                flat_axes[i].plot(epochs, val_values, 'r-', label='Validation')
            
            flat_axes[i].set_title(f'Évolution de {metric.capitalize()}')
            flat_axes[i].set_xlabel('Époques')
            flat_axes[i].set_ylabel(metric.capitalize())
            flat_axes[i].legend()
            flat_axes[i].grid(True)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder la figure
        metrics_path = os.path.join(self.vis_dir, 'metrics_history.png')
        plt.savefig(metrics_path)
        plt.close(fig2)
        
        # Ajouter à TensorBoard
        if self.writer:
            self.writer.add_figure('History/Loss', fig1, 0)
            self.writer.add_figure('History/Metrics', fig2, 0)


class ConfusionVisualizer(VisualizationCallback):
    """
    Callback spécialisé pour visualiser les matrices de confusion.
    
    Ce callback génère des visualisations détaillées des métriques de confusion
    pour analyser les performances du modèle.
    """
    
    def __init__(self, log_dir: str, val_dataset=None, 
                 save_frequency: int = 5, use_tensorboard: bool = True):
        """
        Initialise le visualiseur de confusion.
        
        Args:
            log_dir: Répertoire où sauvegarder les visualisations.
            val_dataset: Dataset de validation pour les visualisations.
            save_frequency: Fréquence de sauvegarde des visualisations (en époques).
            use_tensorboard: Utiliser TensorBoard pour les visualisations.
        """
        super(ConfusionVisualizer, self).__init__(
            log_dir=log_dir, val_dataset=val_dataset,
            save_frequency=save_frequency, use_tensorboard=use_tensorboard
        )
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Appelé à la fin de chaque époque.
        
        Args:
            epoch: Numéro de l'époque actuelle.
            logs: Dictionnaire contenant des informations sur l'état de l'entraînement.
        """
        # Vérifier si on doit générer des visualisations à cette époque
        if (epoch + 1) % self.save_frequency != 0:
            return
        
        if logs is None:
            return
        
        # Récupérer les métriques de confusion
        val_metrics = logs.get('val_metrics', {})
        confusion_data = val_metrics.get('confusion_matrix', None)
        
        if confusion_data:
            self._visualize_confusion_matrix(confusion_data, epoch)
    
    def _visualize_confusion_matrix(self, confusion_data: Dict[str, float], epoch: int) -> None:
        """
        Visualise la matrice de confusion.
        
        Args:
            confusion_data: Données de la matrice de confusion.
            epoch: Numéro de l'époque actuelle.
        """
        # Extraire les valeurs
        tp = confusion_data.get('tp', 0)
        fp = confusion_data.get('fp', 0)
        tn = confusion_data.get('tn', 0)
        fn = confusion_data.get('fn', 0)
        
        # Calculer les taux
        tp_rate = confusion_data.get('tp_rate', tp / (tp + fn) if tp + fn > 0 else 0)
        fp_rate = confusion_data.get('fp_rate', fp / (fp + tn) if fp + tn > 0 else 0)
        tn_rate = confusion_data.get('tn_rate', tn / (tn + fp) if tn + fp > 0 else 0)
        fn_rate = confusion_data.get('fn_rate', fn / (fn + tp) if fn + tp > 0 else 0)
        
        # Créer la figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Matrice de confusion absolue
        confusion_matrix = np.array([[tp, fp], [fn, tn]])
        im = axes[0].imshow(confusion_matrix, cmap='Blues')
        axes[0].set_title('Matrice de confusion (valeurs absolues)')
        axes[0].set_xlabel('Prédit')
        axes[0].set_ylabel('Réel')
        axes[0].set_xticks([0, 1])
        axes[0].set_yticks([0, 1])
        axes[0].set_xticklabels(['Positif', 'Négatif'])
        axes[0].set_yticklabels(['Positif', 'Négatif'])
        
        # Ajouter les valeurs dans les cellules
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, f'{confusion_matrix[i, j]:.0f}',
                           ha='center', va='center', color='black')
        
        # Matrice de confusion normalisée
        confusion_matrix_norm = np.array([[tp_rate, fp_rate], [fn_rate, tn_rate]])
        im = axes[1].imshow(confusion_matrix_norm, cmap='Blues', vmin=0, vmax=1)
        axes[1].set_title('Matrice de confusion (taux)')
        axes[1].set_xlabel('Prédit')
        axes[1].set_ylabel('Réel')
        axes[1].set_xticks([0, 1])
        axes[1].set_yticks([0, 1])
        axes[1].set_xticklabels(['Positif', 'Négatif'])
        axes[1].set_yticklabels(['Positif', 'Négatif'])
        
        # Ajouter les valeurs dans les cellules
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, f'{confusion_matrix_norm[i, j]:.3f}',
                           ha='center', va='center', color='black')
        
        # Barre de couleur
        cbar = fig.colorbar(im, ax=axes[1])
        cbar.set_label('Taux')
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder la figure
        conf_path = os.path.join(self.vis_dir, f'confusion_matrix_epoch_{epoch+1:03d}.png')
        plt.savefig(conf_path)
        plt.close(fig)
        
        # Ajouter à TensorBoard
        if self.writer:
            self.writer.add_figure(f'ConfusionMatrix/Epoch_{epoch+1}', fig, epoch) 