"""
Evaluation metrics and functions for FathomNet baseline model.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
import os

# Import FathomNet for hierarchical distance calculation
try:
    import fathomnet.api.worms as worms
    FATHOMNET_AVAILABLE = True
except ImportError:
    FATHOMNET_AVAILABLE = False
    logging.warning("fathomnet-py not available. Hierarchical distance evaluation disabled.")

from .config import BaselineConfig


class HierarchicalEvaluator:
    """
    Evaluator for hierarchical classification with FathomNet taxonomic distances.
    """
    
    def __init__(self, class_names: List[str], config: BaselineConfig):
        """
        Initialize the evaluator.
        
        Args:
            class_names: List of class names in order
            config: Configuration object
        """
        self.class_names = class_names
        self.config = config
        self.num_classes = len(class_names)
        
        # Cache for taxonomic information
        self._taxonomy_cache = {}
        
        # Initialize WoRMS taxonomic information if available
        if FATHOMNET_AVAILABLE and config.calculate_hierarchical_distance:
            self._load_taxonomic_info()
    
    def _load_taxonomic_info(self):
        """Load taxonomic information for all classes."""
        logging.info("Loading taxonomic information for hierarchical evaluation...")
        
        for class_name in self.class_names:
            try:
                info = worms.get_info(class_name)
                self._taxonomy_cache[class_name] = info
                logging.debug(f"Loaded taxonomy for {class_name}: {info.rank}")
            except Exception as e:
                logging.warning(f"Failed to load taxonomy for {class_name}: {e}")
                # Use a default empty taxonomy
                self._taxonomy_cache[class_name] = None
        
        logging.info(f"Loaded taxonomic information for {len(self._taxonomy_cache)} classes")
    
    def calculate_hierarchical_distance(
        self,
        predicted_names: List[str],
        true_names: List[str]
    ) -> Tuple[float, List[float]]:
        """
        Calculate hierarchical distance between predictions and ground truth.
        
        Args:
            predicted_names: List of predicted class names
            true_names: List of true class names
            
        Returns:
            Tuple of (mean_distance, individual_distances)
        """
        if not FATHOMNET_AVAILABLE:
            logging.warning("FathomNet not available. Returning dummy distances.")
            return 0.0, [0.0] * len(predicted_names)
        
        distances = []
        
        for pred_name, true_name in zip(predicted_names, true_names):
            try:
                # Get taxonomic info from cache
                pred_info = self._taxonomy_cache.get(pred_name)
                true_info = self._taxonomy_cache.get(true_name)
                
                if pred_info is None or true_info is None:
                    # Use maximum penalty for missing taxonomy
                    distance = 12.0
                else:
                    # Calculate hierarchical distance using FathomNet
                    distance = worms.get_distance(pred_info, true_info)
                
                distances.append(distance)
                
            except Exception as e:
                logging.warning(f"Error calculating distance for {pred_name} vs {true_name}: {e}")
                distances.append(12.0)  # Maximum penalty
        
        mean_distance = np.mean(distances)
        return mean_distance, distances
    
    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of predictions.
        
        Args:
            y_true: True class indices
            y_pred: Predicted class indices
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {}
        
        # Basic accuracy metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['top1_accuracy'] = results['accuracy']
        
        # Top-k accuracy if probabilities provided
        if y_proba is not None:
            results['top5_accuracy'] = self._calculate_topk_accuracy(y_true, y_proba, k=5)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        results['per_class_precision'] = precision
        results['per_class_recall'] = recall
        results['per_class_f1'] = f1
        results['per_class_support'] = support
        
        # Macro and weighted averages
        results['macro_precision'] = np.mean(precision)
        results['macro_recall'] = np.mean(recall)
        results['macro_f1'] = np.mean(f1)
        
        results['weighted_precision'] = np.average(precision, weights=support)
        results['weighted_recall'] = np.average(recall, weights=support)
        results['weighted_f1'] = np.average(f1, weights=support)
        
        # Convert indices to class names for hierarchical evaluation
        if self.config.calculate_hierarchical_distance:
            pred_names = [self.class_names[idx] for idx in y_pred]
            true_names = [self.class_names[idx] for idx in y_true]
            
            mean_distance, individual_distances = self.calculate_hierarchical_distance(
                pred_names, true_names
            )
            
            results['hierarchical_distance'] = mean_distance
            results['individual_distances'] = individual_distances
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return results
    
    def _calculate_topk_accuracy(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        k: int = 5
    ) -> float:
        """Calculate top-k accuracy."""
        if k >= self.num_classes:
            return 1.0
        
        # Get top-k predictions
        top_k_pred = np.argsort(y_proba, axis=1)[:, -k:]
        
        # Check if true label is in top-k
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_pred[i]:
                correct += 1
        
        return correct / len(y_true)
    
    def generate_classification_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed classification report."""
        report = []
        report.append("=" * 60)
        report.append("FATHOMNET BASELINE MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        # Overall metrics
        report.append(f"\nOVERALL METRICS:")
        report.append(f"  Top-1 Accuracy: {results['accuracy']:.4f}")
        
        if 'top5_accuracy' in results:
            report.append(f"  Top-5 Accuracy: {results['top5_accuracy']:.4f}")
        
        if 'hierarchical_distance' in results:
            report.append(f"  Hierarchical Distance: {results['hierarchical_distance']:.4f}")
        
        report.append(f"\nAGGREGATED METRICS:")
        report.append(f"  Macro Precision: {results['macro_precision']:.4f}")
        report.append(f"  Macro Recall: {results['macro_recall']:.4f}")
        report.append(f"  Macro F1-Score: {results['macro_f1']:.4f}")
        report.append(f"  Weighted Precision: {results['weighted_precision']:.4f}")
        report.append(f"  Weighted Recall: {results['weighted_recall']:.4f}")
        report.append(f"  Weighted F1-Score: {results['weighted_f1']:.4f}")
        
        # Per-class performance (top 10 best and worst)
        class_f1 = results['per_class_f1']
        class_support = results['per_class_support']
        
        # Sort by F1 score
        sorted_indices = np.argsort(class_f1)[::-1]
        
        report.append(f"\nTOP 10 PERFORMING CLASSES (by F1-score):")
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[i]
            class_name = self.class_names[idx]
            f1 = class_f1[idx]
            support = class_support[idx]
            report.append(f"  {class_name}: F1={f1:.3f}, Support={support}")
        
        report.append(f"\nWORST 10 PERFORMING CLASSES (by F1-score):")
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[-(i+1)]
            class_name = self.class_names[idx]
            f1 = class_f1[idx]
            support = class_support[idx]
            report.append(f"  {class_name}: F1={f1:.3f}, Support={support}")
        
        return "\n".join(report)
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 12)
    ):
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=figsize)
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=False,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_class_performance(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 8)
    ):
        """Plot per-class performance metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # F1 scores
        class_f1 = results['per_class_f1']
        sorted_indices = np.argsort(class_f1)
        
        ax1.barh(range(len(class_f1)), class_f1[sorted_indices])
        ax1.set_yticks(range(len(class_f1)))
        ax1.set_yticklabels([self.class_names[i] for i in sorted_indices])
        ax1.set_xlabel('F1-Score')
        ax1.set_title('Per-Class F1-Score')
        ax1.grid(True, alpha=0.3)
        
        # Support (number of samples per class)
        class_support = results['per_class_support']
        
        ax2.barh(range(len(class_support)), class_support[sorted_indices])
        ax2.set_yticks(range(len(class_support)))
        ax2.set_yticklabels([self.class_names[i] for i in sorted_indices])
        ax2.set_xlabel('Number of Samples')
        ax2.set_title('Per-Class Support')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Class performance plot saved to {save_path}")
        
        plt.show()


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    evaluator: HierarchicalEvaluator,
    save_predictions: bool = True,
    predictions_path: str = "predictions.csv"
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        evaluator: HierarchicalEvaluator instance
        save_predictions: Whether to save predictions to file
        predictions_path: Path to save predictions
        
    Returns:
        Dictionary containing evaluation results
    """
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_metadata = []
    
    with torch.no_grad():
        for batch_idx, (images, labels, metadata) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_metadata.extend(metadata)
            
            if batch_idx % 50 == 0:
                logging.info(f"Evaluated batch {batch_idx}/{len(data_loader)}")
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_probabilities)
    
    # Evaluate predictions
    results = evaluator.evaluate_predictions(y_true, y_pred, y_proba)
    
    # Save predictions if requested
    if save_predictions:
        predictions_df = pd.DataFrame({
            'annotation_id': [meta['annotation_id'] for meta in all_metadata],
            'image_id': [meta['image_id'] for meta in all_metadata],
            'true_class_idx': y_true,
            'predicted_class_idx': y_pred,
            'true_class_name': [evaluator.class_names[idx] for idx in y_true],
            'predicted_class_name': [evaluator.class_names[idx] for idx in y_pred],
            'confidence': np.max(y_proba, axis=1)
        })
        
        # Add probability columns
        for i, class_name in enumerate(evaluator.class_names):
            predictions_df[f'prob_{class_name}'] = y_proba[:, i]
        
        predictions_df.to_csv(predictions_path, index=False)
        logging.info(f"Predictions saved to {predictions_path}")
    
    # Add metadata to results
    results['total_samples'] = len(y_true)
    results['predictions_df'] = predictions_df if save_predictions else None
    
    return results


def create_submission_file(
    predictions_df: pd.DataFrame,
    output_path: str,
    class_names: List[str]
) -> None:
    """
    Create submission file in competition format.
    
    Args:
        predictions_df: DataFrame with predictions
        output_path: Path to save submission file
        class_names: List of class names
    """
    submission_df = pd.DataFrame({
        'annotation_id': predictions_df['annotation_id'],
        'concept_name': predictions_df['predicted_class_name']
    })
    
    submission_df.to_csv(output_path, index=False)
    logging.info(f"Submission file saved to {output_path}")
    logging.info(f"Submission contains {len(submission_df)} predictions")