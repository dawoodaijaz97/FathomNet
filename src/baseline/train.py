"""
Training script for FathomNet baseline model.
"""

import argparse
import os
import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import BaselineConfig, get_config
from .dataset import create_data_loaders, create_test_loader
from .model import create_model, save_checkpoint, load_checkpoint
from .evaluate import HierarchicalEvaluator, evaluate_model, create_submission_file
from .utils import (
    set_seed, setup_logging, get_device, count_parameters,
    AverageMeter, ProgressMeter, EarlyStopping, TrainingLogger,
    Timer, format_time, calculate_class_weights
)


class BaselineTrainer:
    """
    Trainer class for FathomNet baseline model.
    """
    
    def __init__(self, config: BaselineConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = get_device()
        
        # Initialize model, data loaders, and training components
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.evaluator = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.early_stopping = None
        self.logger = None
        
        # Setup
        self._setup_logging()
        self._setup_data()
        self._setup_model()
        self._setup_training()
    
    def _setup_logging(self):
        """Setup logging and experiment tracking."""
        log_file = os.path.join(self.config.save_dir, "training.log")
        setup_logging(log_file=log_file, verbose=self.config.verbose)
        
        # Create training logger
        log_dir = os.path.join(self.config.save_dir, "logs")
        self.logger = TrainingLogger(log_dir=log_dir, use_tensorboard=True)
        
        logging.info("=" * 60)
        logging.info("FATHOMNET BASELINE MODEL TRAINING")
        logging.info("=" * 60)
        logging.info(f"Configuration: {self.config}")
    
    def _setup_data(self):
        """Setup data loaders."""
        logging.info("Setting up data loaders...")
        
        # Create train and validation loaders
        self.train_loader, self.val_loader, self.dataset_info = create_data_loaders(self.config)
        
        logging.info(f"Training samples: {self.dataset_info['train_size']}")
        logging.info(f"Validation samples: {self.dataset_info['val_size']}")
        logging.info(f"Number of classes: {self.dataset_info['num_classes']}")
        
        # Create evaluator
        self.evaluator = HierarchicalEvaluator(
            class_names=self.dataset_info['class_names'],
            config=self.config
        )
        
        # Update config with actual number of classes
        self.config.num_classes = self.dataset_info['num_classes']
    
    def _setup_model(self):
        """Setup model and move to device."""
        logging.info("Setting up model...")
        
        # Create model
        self.model = create_model(self.config)
        self.model.to(self.device)
        
        # Log model information
        total_params, trainable_params = count_parameters(self.model)
        logging.info(f"Model moved to {self.device}")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
    
    def _setup_training(self):
        """Setup training components (optimizer, scheduler, loss, etc.)."""
        logging.info("Setting up training components...")
        
        # Loss function
        if hasattr(self.config, 'use_class_weights') and self.config.use_class_weights:
            # Calculate class weights for imbalanced dataset
            weights = calculate_class_weights(
                self.train_loader.dataset.annotations,
                self.config.num_classes
            )
            weights = weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
            logging.info("Using weighted CrossEntropyLoss")
        else:
            self.criterion = nn.CrossEntropyLoss()
            logging.info("Using standard CrossEntropyLoss")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.optimizer_betas
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Monitor validation accuracy (higher is better)
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
            min_lr=self.config.lr_scheduler_min_lr,
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            verbose=True
        )
        
        logging.info(f"Optimizer: Adam (lr={self.config.learning_rate}, wd={self.config.weight_decay})")
        logging.info(f"Scheduler: ReduceLROnPlateau (patience={self.config.lr_scheduler_patience})")
        logging.info(f"Early stopping: patience={self.config.early_stopping_patience}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        
        # Metrics tracking
        losses = AverageMeter('Loss', ':.4f')
        accuracies = AverageMeter('Acc', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, accuracies],
            prefix=f"Epoch: [{self.current_epoch}]"
        )
        
        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == labels).float().mean().item()
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy * 100, images.size(0))
            
            # Log progress
            if batch_idx % self.config.log_interval == 0:
                progress.display(batch_idx)
        
        return {
            'train_loss': losses.avg,
            'train_acc': accuracies.avg
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        
        losses = AverageMeter('Loss', ':.4f')
        accuracies = AverageMeter('Acc', ':6.2f')
        
        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == labels).float().mean().item()
                
                # Update metrics
                losses.update(loss.item(), images.size(0))
                accuracies.update(accuracy * 100, images.size(0))
        
        return {
            'val_loss': losses.avg,
            'val_acc': accuracies.avg
        }
    
    def train(self):
        """Main training loop."""
        logging.info("Starting training...")
        logging.info(f"Training for {self.config.max_epochs} epochs")
        
        # Check if we should freeze backbone initially
        if self.config.freeze_epochs > 0:
            self.model.freeze_backbone()
            logging.info(f"Backbone frozen for first {self.config.freeze_epochs} epochs")
        
        timer = Timer()
        timer.start()
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            # Unfreeze backbone after freeze_epochs
            if epoch == self.config.freeze_epochs and self.model.backbone_frozen:
                self.model.unfreeze_backbone()
                # Reduce learning rate for fine-tuning
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.config.finetune_lr_factor
                logging.info(f"Backbone unfrozen, LR reduced by factor {self.config.finetune_lr_factor}")
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.logger.log_metrics(epoch, epoch_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_acc'])
            
            # Check for best model
            is_best = val_metrics['val_acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['val_acc']
                logging.info(f"New best validation accuracy: {self.best_val_acc:.4f}%")
            
            # Save checkpoint
            if self.config.save_best_only and is_best:
                checkpoint_path = os.path.join(self.config.save_dir, f"checkpoint_epoch_{epoch}.pth")
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    val_acc=val_metrics['val_acc'],
                    best_val_acc=self.best_val_acc,
                    checkpoint_path=checkpoint_path,
                    is_best=is_best
                )
            
            # Early stopping check
            if self.early_stopping(val_metrics['val_acc'], self.model):
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Training completed
        training_time = timer.stop()
        logging.info(f"Training completed in {format_time(training_time)}")
        logging.info(f"Best validation accuracy: {self.best_val_acc:.4f}%")
        
        # Save final metrics and plots
        metrics_path = os.path.join(self.config.save_dir, "training_metrics.json")
        self.logger.save_metrics(metrics_path)
        
        curves_path = os.path.join(self.config.save_dir, "training_curves.png")
        self.logger.plot_training_curves(curves_path)
        
        self.logger.close()
    
    def evaluate(self, use_best_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            use_best_checkpoint: Whether to load the best checkpoint
            
        Returns:
            Evaluation results
        """
        logging.info("Evaluating model...")
        
        if use_best_checkpoint:
            # Load best checkpoint
            best_checkpoint_path = os.path.join(self.config.save_dir, "checkpoint_best.pth")
            if os.path.exists(best_checkpoint_path):
                load_checkpoint(self.model, best_checkpoint_path, self.device)
                logging.info("Loaded best checkpoint for evaluation")
        
        # Evaluate on validation set
        predictions_path = os.path.join(self.config.save_dir, "validation_predictions.csv")
        results = evaluate_model(
            model=self.model,
            data_loader=self.val_loader,
            device=self.device,
            evaluator=self.evaluator,
            save_predictions=self.config.save_predictions,
            predictions_path=predictions_path
        )
        
        # Generate and save evaluation report
        report = self.evaluator.generate_classification_report(results)
        logging.info("\n" + report)
        
        report_path = os.path.join(self.config.save_dir, "evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save visualizations
        if 'confusion_matrix' in results:
            cm_path = os.path.join(self.config.save_dir, "confusion_matrix.png")
            self.evaluator.plot_confusion_matrix(
                results['confusion_matrix'],
                save_path=cm_path
            )
        
        perf_path = os.path.join(self.config.save_dir, "class_performance.png")
        self.evaluator.plot_class_performance(results, save_path=perf_path)
        
        return results
    
    def predict_test(self) -> str:
        """
        Generate predictions for test set.
        
        Returns:
            Path to submission file
        """
        logging.info("Generating test predictions...")
        
        # Create test loader if not already created
        if self.test_loader is None:
            self.test_loader = create_test_loader(self.config, self.dataset_info)
        
        # Generate predictions
        predictions_path = os.path.join(self.config.save_dir, "test_predictions.csv")
        results = evaluate_model(
            model=self.model,
            data_loader=self.test_loader,
            device=self.device,
            evaluator=self.evaluator,
            save_predictions=True,
            predictions_path=predictions_path
        )
        
        # Create submission file
        submission_path = os.path.join(self.config.save_dir, "submission.csv")
        create_submission_file(
            predictions_df=results['predictions_df'],
            output_path=submission_path,
            class_names=self.dataset_info['class_names']
        )
        
        logging.info(f"Test predictions saved to {predictions_path}")
        logging.info(f"Submission file saved to {submission_path}")
        
        return submission_path


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train FathomNet Baseline Model')
    
    # Configuration arguments
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--model', type=str, default='resnet50', help='Model architecture')
    
    # Data arguments
    parser.add_argument('--train-json', type=str, help='Path to training JSON file')
    parser.add_argument('--test-json', type=str, help='Path to test JSON file')
    parser.add_argument('--images-dir', type=str, help='Path to images directory')
    
    # Training arguments
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--evaluate-only', action='store_true', help='Only evaluate, no training')
    parser.add_argument('--predict-test', action='store_true', help='Generate test predictions')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create configuration
    config_overrides = {}
    if args.batch_size: config_overrides['batch_size'] = args.batch_size
    if args.learning_rate: config_overrides['learning_rate'] = args.learning_rate
    if args.epochs: config_overrides['max_epochs'] = args.epochs
    if args.model: config_overrides['model_name'] = args.model
    if args.train_json: config_overrides['train_json_path'] = args.train_json
    if args.test_json: config_overrides['test_json_path'] = args.test_json
    if args.images_dir: config_overrides['images_dir'] = args.images_dir
    if args.save_dir: config_overrides['save_dir'] = args.save_dir
    if args.verbose: config_overrides['verbose'] = args.verbose
    
    config = get_config(**config_overrides)
    
    # Create trainer
    trainer = BaselineTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        load_checkpoint(trainer.model, args.resume, trainer.device)
        logging.info(f"Resumed from checkpoint: {args.resume}")
    
    # Run training or evaluation
    if args.evaluate_only:
        results = trainer.evaluate()
        logging.info(f"Evaluation completed. Best accuracy: {results['accuracy']:.4f}")
    elif args.predict_test:
        submission_path = trainer.predict_test()
        logging.info(f"Test predictions completed. Submission saved to: {submission_path}")
    else:
        # Full training pipeline
        trainer.train()
        results = trainer.evaluate()
        submission_path = trainer.predict_test()
        
        logging.info("Training pipeline completed successfully!")
        logging.info(f"Best validation accuracy: {trainer.best_val_acc:.4f}%")
        logging.info(f"Final evaluation accuracy: {results['accuracy']:.4f}")
        if 'hierarchical_distance' in results:
            logging.info(f"Hierarchical distance: {results['hierarchical_distance']:.4f}")
        logging.info(f"Submission file: {submission_path}")


if __name__ == "__main__":
    main()