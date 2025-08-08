"""
Configuration settings for the baseline model.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class BaselineConfig:
    """Configuration class for baseline model training and evaluation."""
    
    # Data paths
    train_json_path: str = "../../datasets/dataset_train.json"
    test_json_path: str = "../../datasets/dataset_test.json"
    images_dir: str = "../../datasets/images"
    
    # Model settings
    model_name: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 79
    
    # Input settings
    input_size: Tuple[int, int] = (224, 224)
    normalize_mean: List[float] = [0.485, 0.456, 0.406]
    normalize_std: List[float] = [0.229, 0.224, 0.225]
    
    # Training settings
    batch_size: int = 32
    num_workers: int = 4
    max_epochs: int = 50
    
    # Optimization settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer_betas: Tuple[float, float] = (0.9, 0.999)
    
    # Learning rate schedule
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    lr_scheduler_min_lr: float = 1e-7
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Fine-tuning strategy
    freeze_epochs: int = 5
    finetune_lr_factor: float = 0.1
    
    # Data augmentation
    horizontal_flip_prob: float = 0.5
    rotation_degrees: int = 15
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1
    
    # Validation split
    validation_split: float = 0.2
    stratify: bool = True
    random_seed: int = 42
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    save_best_only: bool = True
    monitor_metric: str = "val_accuracy"
    
    # Logging
    log_interval: int = 10
    verbose: bool = True
    
    # Evaluation
    calculate_hierarchical_distance: bool = True
    save_predictions: bool = True
    
    def __post_init__(self):
        """Post-initialization to create directories and validate settings."""
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Validate settings
        assert 0 < self.validation_split < 1, "Validation split must be between 0 and 1"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.num_classes > 0, "Number of classes must be positive"


# Default configuration instance
default_config = BaselineConfig()


def get_config(**kwargs) -> BaselineConfig:
    """
    Get configuration with optional overrides.
    
    Args:
        **kwargs: Configuration overrides
        
    Returns:
        BaselineConfig: Configuration instance with applied overrides
    """
    config_dict = default_config.__dict__.copy()
    config_dict.update(kwargs)
    return BaselineConfig(**config_dict)