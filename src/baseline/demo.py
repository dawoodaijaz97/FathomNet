#!/usr/bin/env python3
"""
Demo script for FathomNet baseline model.

This script demonstrates how to use the baseline model for training and evaluation.
"""

import os
import sys
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.config import BaselineConfig
from baseline.train import BaselineTrainer
from baseline.utils import set_seed, setup_logging


def demo_training():
    """Demonstrate training the baseline model."""
    print("=" * 60)
    print("FATHOMNET BASELINE MODEL DEMO")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Setup basic logging
    setup_logging(verbose=True)
    
    # Create configuration with demo settings
    config = BaselineConfig(
        # Data paths (adjust these to your actual paths)
        train_json_path="../../datasets/dataset_train.json",
        test_json_path="../../datasets/dataset_test.json", 
        images_dir="../../datasets/images",
        
        # Model settings
        model_name="resnet50",
        pretrained=True,
        
        # Training settings (reduced for demo)
        batch_size=16,  # Smaller batch size for demo
        max_epochs=5,   # Few epochs for demo
        learning_rate=1e-4,
        
        # Validation
        validation_split=0.2,
        
        # Paths
        save_dir="./demo_output",
        
        # Logging
        verbose=True,
        log_interval=5
    )
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Save directory: {config.save_dir}")
    
    try:
        # Create trainer
        print(f"\nCreating trainer...")
        trainer = BaselineTrainer(config)
        
        # Run training
        print(f"\nStarting training...")
        trainer.train()
        
        # Evaluate model
        print(f"\nEvaluating model...")
        results = trainer.evaluate()
        
        # Generate test predictions
        print(f"\nGenerating test predictions...")
        submission_path = trainer.predict_test()
        
        # Print summary
        print(f"\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Best validation accuracy: {trainer.best_val_acc:.4f}%")
        print(f"Final evaluation accuracy: {results['accuracy']:.4f}")
        if 'hierarchical_distance' in results:
            print(f"Hierarchical distance: {results['hierarchical_distance']:.4f}")
        print(f"Submission file saved to: {submission_path}")
        print(f"All outputs saved to: {config.save_dir}")
        
        return True
        
    except Exception as e:
        logging.error(f"Demo failed with error: {e}")
        print(f"\nDemo failed. Check the error message above.")
        print(f"Common issues:")
        print(f"  - Make sure dataset paths are correct")
        print(f"  - Ensure you have downloaded the competition data")
        print(f"  - Check that all dependencies are installed")
        return False


def demo_config_customization():
    """Demonstrate different configuration options."""
    print("\n" + "=" * 60)
    print("CONFIGURATION CUSTOMIZATION EXAMPLES")
    print("=" * 60)
    
    # Example 1: High-performance configuration
    print("\n1. High-performance configuration (for full training):")
    config_hp = BaselineConfig(
        model_name="resnet101",
        batch_size=64,
        max_epochs=100,
        learning_rate=1e-4,
        freeze_epochs=10,
        early_stopping_patience=15,
        save_dir="./high_performance_output"
    )
    print(f"   Model: {config_hp.model_name}")
    print(f"   Batch size: {config_hp.batch_size}")
    print(f"   Max epochs: {config_hp.max_epochs}")
    print(f"   Freeze epochs: {config_hp.freeze_epochs}")
    
    # Example 2: Fast prototyping configuration
    print("\n2. Fast prototyping configuration:")
    config_fast = BaselineConfig(
        model_name="resnet50",
        batch_size=32,
        max_epochs=20,
        learning_rate=1e-3,
        validation_split=0.1,
        early_stopping_patience=5,
        save_dir="./fast_prototype_output"
    )
    print(f"   Model: {config_fast.model_name}")
    print(f"   Batch size: {config_fast.batch_size}")
    print(f"   Max epochs: {config_fast.max_epochs}")
    print(f"   Learning rate: {config_fast.learning_rate}")
    
    # Example 3: EfficientNet configuration
    print("\n3. EfficientNet configuration:")
    config_effnet = BaselineConfig(
        model_name="efficientnet_b3",
        batch_size=48,
        max_epochs=50,
        learning_rate=5e-5,
        input_size=(300, 300),  # Larger input for EfficientNet
        save_dir="./efficientnet_output"
    )
    print(f"   Model: {config_effnet.model_name}")
    print(f"   Input size: {config_effnet.input_size}")
    print(f"   Batch size: {config_effnet.batch_size}")


def demo_command_line_usage():
    """Show command line usage examples."""
    print("\n" + "=" * 60)
    print("COMMAND LINE USAGE EXAMPLES")
    print("=" * 60)
    
    print("\n1. Basic training:")
    print("   python -m baseline.train")
    
    print("\n2. Custom hyperparameters:")
    print("   python -m baseline.train --batch-size 64 --learning-rate 1e-3 --epochs 100")
    
    print("\n3. Different model architecture:")
    print("   python -m baseline.train --model resnet101 --batch-size 32")
    
    print("\n4. Custom data paths:")
    print("   python -m baseline.train --train-json /path/to/train.json --images-dir /path/to/images")
    
    print("\n5. Evaluation only:")
    print("   python -m baseline.train --evaluate-only --resume /path/to/checkpoint.pth")
    
    print("\n6. Generate test predictions only:")
    print("   python -m baseline.train --predict-test --resume /path/to/best_checkpoint.pth")
    
    print("\n7. Full training with custom settings:")
    print("   python -m baseline.train \\")
    print("       --batch-size 64 \\")
    print("       --learning-rate 1e-4 \\")
    print("       --epochs 50 \\")
    print("       --model resnet50 \\")
    print("       --save-dir ./my_experiment \\")
    print("       --verbose")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FathomNet Baseline Model Demo')
    parser.add_argument('--run-training', action='store_true', 
                       help='Run the full training demo (requires dataset)')
    parser.add_argument('--show-configs', action='store_true',
                       help='Show configuration examples')
    parser.add_argument('--show-cli', action='store_true',
                       help='Show command line usage examples')
    
    args = parser.parse_args()
    
    if args.run_training:
        success = demo_training()
        sys.exit(0 if success else 1)
    elif args.show_configs:
        demo_config_customization()
    elif args.show_cli:
        demo_command_line_usage()
    else:
        # Show all demos by default
        print("FathomNet Baseline Model Demo Script")
        print("\nAvailable demos:")
        print("  --run-training  : Run full training demo (requires dataset)")
        print("  --show-configs  : Show configuration examples")
        print("  --show-cli      : Show command line usage examples")
        print("\nTo see all examples:")
        demo_config_customization()
        demo_command_line_usage()
        print(f"\nTo run the training demo:")
        print(f"  python demo.py --run-training")
        print(f"\nNote: Make sure you have downloaded the FathomNet dataset first!")