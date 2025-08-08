"""
Utility functions for FathomNet baseline model training and evaluation.
"""

import os
import logging
import random
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import json
import matplotlib.pyplot as plt
import seaborn as sns

from .config import BaselineConfig


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")


def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    verbose: bool = True
):
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional log file path
        log_level: Logging level
        verbose: Whether to log to console
    """
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")


def get_device(use_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        use_cuda: Whether to use CUDA if available
        
    Returns:
        PyTorch device
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        logging.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU device")
    
    return device


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        """
        Initialize the meter.
        
        Args:
            name: Name of the metric
            fmt: Format string for display
        """
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics.
        
        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Display progress during training."""
    
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        """
        Initialize progress meter.
        
        Args:
            num_batches: Total number of batches
            meters: List of meters to track
            prefix: Prefix for display
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int):
        """Display current progress."""
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(m) for m in self.meters]
        logging.info('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches: int) -> str:
        """Get format string for batch display."""
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class EarlyStopping:
    """Early stopping utility to stop training when validation metric stops improving."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score (higher is better)
            model: Model to save weights from
            
        Returns:
            Whether to stop training
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        logging.info('Restored best weights')
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, model: nn.Module):
        """Save current best model weights."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


class TrainingLogger:
    """Logger for training metrics and visualization."""
    
    def __init__(self, log_dir: str = "./logs", use_tensorboard: bool = True):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save logs
            use_tensorboard: Whether to use TensorBoard
        """
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        os.makedirs(log_dir, exist_ok=True)
        
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics for current epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric values
        """
        # Store metrics
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Log to TensorBoard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)
        
        # Log to console
        metric_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logging.info(f"Epoch {epoch:3d} | {metric_str}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.metrics['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.metrics['train_acc'], label='Train Accuracy')
        axes[0, 1].plot(self.metrics['val_acc'], label='Validation Accuracy')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.metrics['learning_rate'], label='Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Loss vs Accuracy scatter
        axes[1, 1].scatter(self.metrics['val_loss'], self.metrics['val_acc'], alpha=0.7)
        axes[1, 1].set_title('Validation Loss vs Accuracy')
        axes[1, 1].set_xlabel('Validation Loss')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def save_metrics(self, save_path: str):
        """Save metrics to JSON file."""
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logging.info(f"Metrics saved to {save_path}")
    
    def close(self):
        """Close logger and release resources."""
        if self.writer:
            self.writer.close()


def load_config_from_args(args: Any) -> BaselineConfig:
    """
    Load configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration object
    """
    config_dict = {}
    
    # Map common argument names to config attributes
    arg_mapping = {
        'batch_size': 'batch_size',
        'learning_rate': 'learning_rate',
        'epochs': 'max_epochs',
        'model': 'model_name',
        'data_dir': 'images_dir'
    }
    
    for arg_name, config_attr in arg_mapping.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            config_dict[config_attr] = getattr(args, arg_name)
    
    return BaselineConfig(**config_dict)


def save_model_summary(model: nn.Module, config: BaselineConfig, save_path: str):
    """
    Save model summary to file.
    
    Args:
        model: PyTorch model
        config: Configuration object
        save_path: Path to save summary
    """
    total_params, trainable_params = count_parameters(model)
    
    summary = {
        'model_name': config.model_name,
        'num_classes': config.num_classes,
        'input_size': config.input_size,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'config': config.__dict__
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Model summary saved to {save_path}")


def calculate_class_weights(
    annotations: List[Dict[str, Any]],
    num_classes: int,
    method: str = 'balanced'
) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        annotations: List of annotation dictionaries
        num_classes: Number of classes
        method: Method to calculate weights ('balanced' or 'log')
        
    Returns:
        Tensor of class weights
    """
    # Count samples per class
    class_counts = torch.zeros(num_classes)
    for ann in annotations:
        class_counts[ann['category_id']] += 1
    
    if method == 'balanced':
        # Inverse frequency weighting
        total_samples = len(annotations)
        weights = total_samples / (num_classes * class_counts)
    elif method == 'log':
        # Log-based weighting
        weights = torch.log(class_counts.max() / class_counts)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Handle zero counts
    weights[class_counts == 0] = 0.0
    
    logging.info(f"Calculated class weights using {method} method")
    logging.info(f"Weight range: {weights.min():.3f} - {weights.max():.3f}")
    
    return weights


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


class Timer:
    """Simple timer utility."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time if self.end_time else time.time()
        return end_time - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()