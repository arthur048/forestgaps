# Module d'intégration avec TensorBoard
"""
Fonctions d'intégration avec TensorBoard pour ForestGaps.

Ce module fournit des fonctions pour visualiser les données, les résultats
et les métriques du workflow ForestGaps dans TensorBoard.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union, Any


def add_metrics_to_tensorboard(writer, metrics, epoch, prefix=''):
    """
    Ajoute les métriques à TensorBoard.
    
    Args:
        writer: SummaryWriter de TensorBoard.
        metrics (dict): Dictionnaire de métriques.
        epoch (int): Époque actuelle.
        prefix (str): Préfixe pour les noms des métriques.
    """
    # Ajouter les métriques globales
    for metric_name, value in metrics.items():
        if metric_name != 'threshold_metrics' and metric_name != 'confusion_data':
            writer.add_scalar(f'{prefix}{metric_name}', value, epoch)
    
    # Ajouter les métriques par seuil
    if 'threshold_metrics' in metrics and metrics['threshold_metrics']:
        for threshold, threshold_metrics in metrics['threshold_metrics'].items():
            for metric_name, value in threshold_metrics.items():
                writer.add_scalar(f'{prefix}threshold_{threshold}/{metric_name}', value, epoch)


def add_confusion_matrix_to_tensorboard(writer, confusion_data, epoch, prefix=''):
    """
    Ajoute la matrice de confusion à TensorBoard.
    
    Args:
        writer: SummaryWriter de TensorBoard.
        confusion_data (dict): Dictionnaire contenant les données de confusion.
        epoch (int): Époque actuelle.
        prefix (str): Préfixe pour les noms des métriques.
    """
    # Extraire les données
    matrix = confusion_data['matrix']
    
    # Ajouter les valeurs individuelles
    for key, value in matrix.items():
        writer.add_scalar(f'{prefix}confusion/{key}', value, epoch)
    
    # Ajouter les pourcentages si disponibles
    if 'percentages' in confusion_data:
        for key, value in confusion_data['percentages'].items():
            writer.add_scalar(f'{prefix}confusion_percent/{key}', value * 100, epoch)


def add_model_graph_to_tensorboard(writer, model, input_size=(1, 1, 256, 256), threshold_size=(1, 1)):
    """
    Ajoute le graphe du modèle à TensorBoard.
    
    Args:
        writer: SummaryWriter de TensorBoard.
        model: Modèle PyTorch.
        input_size (tuple): Taille de l'entrée du modèle.
        threshold_size (tuple): Taille du tenseur de seuil.
    """
    # Créer des tenseurs d'exemple
    device = next(model.parameters()).device
    dummy_input = (
        torch.zeros(input_size, device=device),
        torch.zeros(threshold_size, device=device)
    )
    
    # Ajouter le graphe
    try:
        writer.add_graph(model, dummy_input)
    except Exception as e:
        print(f"Erreur lors de l'ajout du graphe du modèle à TensorBoard: {str(e)}")


def visualize_predictions_tensorboard(model, dataloader, device, writer, epoch, threshold_value, num_samples=4):
    """
    Visualise les prédictions du modèle dans TensorBoard.
    
    Args:
        model: Modèle PyTorch.
        dataloader: DataLoader contenant les données.
        device: Périphérique (CPU ou GPU).
        writer: SummaryWriter de TensorBoard.
        epoch (int): Époque actuelle.
        threshold_value (float): Seuil de hauteur pour la segmentation.
        num_samples (int): Nombre d'échantillons à visualiser.
    """
    from forestgaps.training.metrics.segmentation import iou_metric
    
    model.eval()
    
    # Collecter les échantillons
    samples = []
    
    with torch.no_grad():
        for dsm, threshold, target in dataloader:
            if len(samples) >= num_samples:
                break
            
            # Prendre seulement le nombre d'échantillons nécessaires
            for i in range(min(dsm.size(0), num_samples - len(samples))):
                # Vérifier si ce seuil correspond à celui demandé
                threshold_val = threshold[i].item() * max(dataloader.dataset.thresholds)
                closest_threshold = min(dataloader.dataset.thresholds, key=lambda x: abs(x - threshold_val))
                
                if abs(closest_threshold - threshold_value) > 1.0:  # Tolérance de 1m
                    continue
                
                # Prédiction pour cet échantillon
                dsm_i = dsm[i:i+1].to(device)
                threshold_i = threshold[i:i+1].to(device)
                target_i = target[i:i+1].to(device)
                
                # Prédiction
                output_i = model(dsm_i, threshold_i)
                
                # Calculer les métriques pour cet échantillon
                iou = iou_metric(output_i, target_i).item()
                
                # Ajouter aux échantillons
                samples.append({
                    'dsm': dsm_i.cpu(),
                    'target': target_i.cpu(),
                    'output': output_i.cpu(),
                    'threshold_val': threshold_val,
                    'iou': iou
                })
    
    # Créer une grille d'images pour TensorBoard
    if samples:
        # Préparer les images DSM (convertir en RGB pour la visualisation)
        dsm_grid = []
        for sample in samples:
            # Normaliser pour la visualisation
            dsm_img = sample['dsm'][0]  # Forme [C, H, W]
            dsm_img = dsm_img.repeat(3, 1, 1)  # Convertir en RGB en répétant le canal
            dsm_grid.append(dsm_img)
        
        # Préparer les images de masque
        target_grid = []
        for sample in samples:
            # Convertir en RGB (blanc = trouée, noir = non-trouée)
            target_img = sample['target'][0].repeat(3, 1, 1)
            target_grid.append(target_img)
        
        # Préparer les images de prédiction (probabilités)
        prob_grid = []
        for sample in samples:
            # Appliquer une colormap pour une meilleure visualisation des probabilités
            prob = torch.sigmoid(sample['output'][0])
            # Créer une image RGB avec le canal rouge pour les probabilités élevées
            prob_rgb = torch.zeros(3, prob.size(1), prob.size(2))
            prob_rgb[0] = prob  # Canal rouge
            prob_grid.append(prob_rgb)
        
        # Préparer les images de prédiction binaire
        pred_grid = []
        for sample in samples:
            # Convertir en RGB
            pred_binary = (torch.sigmoid(sample['output'][0]) > 0.5).float()
            pred_img = pred_binary.repeat(3, 1, 1)
            pred_grid.append(pred_img)
        
        # Fusionner en un seul tenseur pour chaque type
        dsm_tensor = torch.stack(dsm_grid)
        target_tensor = torch.stack(target_grid)
        prob_tensor = torch.stack(prob_grid)
        pred_tensor = torch.stack(pred_grid)
        
        # Écrire dans TensorBoard
        writer.add_images(f'DSM/threshold_{threshold_value}m', dsm_tensor, epoch)
        writer.add_images(f'Masks_True/threshold_{threshold_value}m', target_tensor, epoch)
        writer.add_images(f'Probabilities/threshold_{threshold_value}m', prob_tensor, epoch)
        writer.add_images(f'Predictions/threshold_{threshold_value}m', pred_tensor, epoch)
        
        # Calculer les métriques moyennes
        avg_metrics = {
            'iou': np.mean([sample['iou'] for sample in samples])
        }
        
        # Ajouter un résumé des métriques
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.text(0.5, 0.5, '\n'.join([f"{m.capitalize()}: {v:.4f}" for m, v in avg_metrics.items()]),
                ha='center', va='center', fontsize=14)
        
        # Convertir la figure en image pour TensorBoard
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        writer.add_images(f'Metrics/threshold_{threshold_value}m', img_tensor, epoch)
        plt.close(fig)


class MonitoringSystem:
    """Centralise le monitoring et la visualisation."""
    
    def __init__(self, log_dir=None, config=None):
        """
        Initialise le système de monitoring.
        
        Args:
            log_dir (str): Répertoire pour les logs TensorBoard.
            config: Configuration du projet.
        """
        self.log_dir = log_dir
        self.config = config
        self.writer = None
        self.epoch = 0
        
        # Initialiser le writer TensorBoard si un répertoire est spécifié
        if log_dir:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
    
    def log_config(self, config):
        """
        Enregistre la configuration dans TensorBoard.
        
        Args:
            config: Configuration du projet.
        """
        if self.writer:
            # Convertir la configuration en texte
            config_text = str(config)
            
            # Ajouter le texte à TensorBoard
            self.writer.add_text('Configuration', config_text)
    
    def log_model_graph(self, model):
        """
        Ajoute le graphe du modèle à TensorBoard.
        
        Args:
            model: Modèle PyTorch.
        """
        if self.writer:
            add_model_graph_to_tensorboard(self.writer, model)
    
    def log_metrics(self, metrics, phase="train"):
        """
        Enregistre les métriques dans TensorBoard.
        
        Args:
            metrics (dict): Dictionnaire de métriques.
            phase (str): Phase d'entraînement ('train' ou 'val').
        """
        if self.writer:
            add_metrics_to_tensorboard(self.writer, metrics, self.epoch, prefix=f'{phase}/')
    
    def log_threshold_metrics(self, metrics_by_threshold):
        """
        Enregistre les métriques par seuil dans TensorBoard.
        
        Args:
            metrics_by_threshold (dict): Dictionnaire de métriques par seuil.
        """
        if self.writer:
            for threshold, metrics in metrics_by_threshold.items():
                for metric_name, value in metrics.items():
                    self.writer.add_scalar(f'threshold_{threshold}/{metric_name}', value, self.epoch)
    
    def log_images(self, dsm, target, output, threshold, phase="val", max_images=6):
        """
        Enregistre des exemples de prédictions dans TensorBoard.
        
        Args:
            dsm: Tenseur DSM.
            target: Tenseur cible.
            output: Tenseur de sortie du modèle.
            threshold: Seuil de hauteur.
            phase (str): Phase d'entraînement.
            max_images (int): Nombre maximum d'images à enregistrer.
        """
        if self.writer:
            # Limiter le nombre d'images
            batch_size = min(dsm.size(0), max_images)
            
            # Préparer les images
            dsm_images = dsm[:batch_size].repeat(1, 3, 1, 1)  # Convertir en RGB
            target_images = target[:batch_size].repeat(1, 3, 1, 1)
            
            # Prédictions (probabilités)
            prob = torch.sigmoid(output[:batch_size])
            prob_images = torch.zeros(batch_size, 3, prob.size(2), prob.size(3))
            prob_images[:, 0] = prob.squeeze(1)  # Canal rouge
            
            # Prédictions binaires
            pred_binary = (prob > 0.5).float()
            pred_images = pred_binary.repeat(1, 3, 1, 1)
            
            # Ajouter à TensorBoard
            self.writer.add_images(f'{phase}/DSM/threshold_{threshold}m', dsm_images, self.epoch)
            self.writer.add_images(f'{phase}/Target/threshold_{threshold}m', target_images, self.epoch)
            self.writer.add_images(f'{phase}/Probabilities/threshold_{threshold}m', prob_images, self.epoch)
            self.writer.add_images(f'{phase}/Predictions/threshold_{threshold}m', pred_images, self.epoch)
    
    def log_resource_usage(self):
        """Enregistre l'utilisation des ressources système."""
        if self.writer:
            import psutil
            import torch
            
            # CPU
            cpu_percent = psutil.cpu_percent()
            self.writer.add_scalar('Resources/CPU_Percent', cpu_percent, self.epoch)
            
            # RAM
            ram_percent = psutil.virtual_memory().percent
            self.writer.add_scalar('Resources/RAM_Percent', ram_percent, self.epoch)
            
            # GPU si disponible
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_mem_alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)  # En GB
                    gpu_mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # En GB
                    
                    self.writer.add_scalar(f'Resources/GPU{i}_Memory_Allocated_GB', gpu_mem_alloc, self.epoch)
                    self.writer.add_scalar(f'Resources/GPU{i}_Memory_Reserved_GB', gpu_mem_reserved, self.epoch)
    
    def log_confusion_metrics(self, pred, target):
        """
        Enregistre les métriques de confusion (FP, FN, etc.).
        
        Args:
            pred: Tenseur de prédictions.
            target: Tenseur cible.
        """
        if self.writer:
            # Calculer la matrice de confusion
            pred_binary = (torch.sigmoid(pred) > 0.5).float()
            
            tp = ((pred_binary == 1) & (target == 1)).sum().item()
            fp = ((pred_binary == 1) & (target == 0)).sum().item()
            tn = ((pred_binary == 0) & (target == 0)).sum().item()
            fn = ((pred_binary == 0) & (target == 1)).sum().item()
            
            total = tp + fp + tn + fn
            
            # Calculer les pourcentages
            tp_percent = tp / total if total > 0 else 0
            fp_percent = fp / total if total > 0 else 0
            tn_percent = tn / total if total > 0 else 0
            fn_percent = fn / total if total > 0 else 0
            
            # Ajouter à TensorBoard
            self.writer.add_scalar('Confusion/TP', tp, self.epoch)
            self.writer.add_scalar('Confusion/FP', fp, self.epoch)
            self.writer.add_scalar('Confusion/TN', tn, self.epoch)
            self.writer.add_scalar('Confusion/FN', fn, self.epoch)
            
            self.writer.add_scalar('Confusion_Percent/TP', tp_percent * 100, self.epoch)
            self.writer.add_scalar('Confusion_Percent/FP', fp_percent * 100, self.epoch)
            self.writer.add_scalar('Confusion_Percent/TN', tn_percent * 100, self.epoch)
            self.writer.add_scalar('Confusion_Percent/FN', fn_percent * 100, self.epoch)
    
    def increment_epoch(self):
        """Incrémente le compteur d'époque."""
        self.epoch += 1
    
    def close(self):
        """Ferme le writer TensorBoard."""
        if self.writer:
            self.writer.close()
