"""
Model implementation for FathomNet baseline.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Optional
import logging

from .config import BaselineConfig


class BaselineModel(nn.Module):
    """
    Baseline model for FathomNet competition using pre-trained ResNet50.
    """
    
    def __init__(self, config: BaselineConfig):
        """
        Initialize the baseline model.
        
        Args:
            config: Configuration object containing model parameters
        """
        super(BaselineModel, self).__init__()
        
        self.config = config
        self.num_classes = config.num_classes
        
        # Load pre-trained backbone
        self.backbone = self._load_backbone()
        
        # Get the feature dimension from the backbone
        self.feature_dim = self._get_feature_dim()
        
        # Create custom classifier
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)
        
        # Initialize classifier weights
        self._init_classifier()
        
        # Track freezing state
        self.backbone_frozen = False
        
    def _load_backbone(self) -> nn.Module:
        """Load and prepare the backbone model."""
        if self.config.model_name.lower() == 'resnet50':
            backbone = models.resnet50(pretrained=self.config.pretrained)
            # Remove the final classification layer
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif self.config.model_name.lower() == 'resnet101':
            backbone = models.resnet101(pretrained=self.config.pretrained)
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif self.config.model_name.lower() == 'efficientnet_b3':
            backbone = models.efficientnet_b3(pretrained=self.config.pretrained)
            # Remove classifier
            backbone.classifier = nn.Identity()
        elif self.config.model_name.lower() == 'efficientnet_b4':
            backbone = models.efficientnet_b4(pretrained=self.config.pretrained)
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")
        
        return backbone
    
    def _get_feature_dim(self) -> int:
        """Get the feature dimension of the backbone."""
        if 'resnet' in self.config.model_name.lower():
            return 2048  # ResNet50/101 feature dimension
        elif 'efficientnet_b3' in self.config.model_name.lower():
            return 1536  # EfficientNet-B3 feature dimension
        elif 'efficientnet_b4' in self.config.model_name.lower():
            return 1792  # EfficientNet-B4 feature dimension
        else:
            # Dynamically determine feature dimension
            dummy_input = torch.randn(1, 3, *self.config.input_size)
            with torch.no_grad():
                features = self.backbone(dummy_input)
                return features.view(features.size(0), -1).size(1)
    
    def _init_classifier(self):
        """Initialize classifier weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone_frozen = True
        logging.info("Backbone frozen for transfer learning")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone_frozen = False
        logging.info("Backbone unfrozen for fine-tuning")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone without classification.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Features of shape (batch_size, feature_dim)
        """
        with torch.no_grad():
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
        return features
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Predicted class indices of shape (batch_size,)
        """
        logits = self.forward(x)
        predictions = torch.argmax(logits, dim=1)
        return predictions


def create_model(config: BaselineConfig) -> BaselineModel:
    """
    Create and initialize the baseline model.
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized baseline model
    """
    model = BaselineModel(config)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Created {config.model_name} baseline model")
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Feature dimension: {model.feature_dim}")
    logging.info(f"Number of classes: {model.num_classes}")
    
    return model


def load_checkpoint(
    model: BaselineModel,
    checkpoint_path: str,
    device: torch.device,
    load_optimizer: bool = False
) -> Dict[str, Any]:
    """
    Load model from checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to map checkpoint to
        load_optimizer: Whether to return optimizer state
        
    Returns:
        Dictionary containing checkpoint information
    """
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logging.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    logging.info(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'unknown'):.4f}")
    
    return checkpoint


def save_checkpoint(
    model: BaselineModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    val_acc: float,
    best_val_acc: float,
    checkpoint_path: str,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state
        epoch: Current epoch number
        val_acc: Current validation accuracy
        best_val_acc: Best validation accuracy so far
        checkpoint_path: Path to save checkpoint
        is_best: Whether this is the best checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'best_val_acc': best_val_acc,
        'config': model.config.__dict__
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = checkpoint_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        logging.info(f"Saved best checkpoint to {best_path}")
    
    logging.info(f"Saved checkpoint to {checkpoint_path}")


class ModelEnsemble(nn.Module):
    """Simple ensemble of multiple models for improved performance."""
    
    def __init__(self, models: list):
        """
        Initialize model ensemble.
        
        Args:
            models: List of trained models to ensemble
        """
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Averaged predictions from all models
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get ensemble prediction probabilities."""
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get ensemble class predictions."""
        logits = self.forward(x)
        predictions = torch.argmax(logits, dim=1)
        return predictions