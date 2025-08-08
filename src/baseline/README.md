# FathomNet Baseline Model

This package contains the baseline model implementation for the FathomNet FGVC 2025 competition as outlined in the [baseline model plan](../../docs/PLAN_base_line_model.md).

## Overview

The baseline model provides a simple but effective starting point for the competition using:
- **ResNet50** pre-trained on ImageNet as the backbone
- **Standard cross-entropy loss** (ignoring taxonomic hierarchy initially) 
- **ROI-based training** using cropped bounding boxes
- **Transfer learning** with initial backbone freezing
- **Comprehensive evaluation** including hierarchical distance metrics

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn pycocotools fathomnet-py tensorboard
```

### 2. Basic Training

```bash
cd src/baseline
python train.py
```

### 3. Custom Training

```bash
python train.py --batch-size 64 --learning-rate 1e-4 --epochs 50 --model resnet50
```

## Package Structure

```
src/baseline/
├── __init__.py          # Package initialization
├── config.py            # Configuration management
├── dataset.py           # COCO data loading and preprocessing
├── model.py             # Model architecture and utilities
├── evaluate.py          # Evaluation metrics and hierarchical distance
├── utils.py             # Training utilities and helpers
├── train.py             # Main training script
├── demo.py              # Demo and example usage
└── README.md            # This file
```

## Configuration

The model is highly configurable through the `BaselineConfig` class:

```python
from baseline.config import BaselineConfig

config = BaselineConfig(
    # Model settings
    model_name="resnet50",
    num_classes=79,
    
    # Training settings
    batch_size=32,
    max_epochs=50,
    learning_rate=1e-4,
    
    # Data settings
    validation_split=0.2,
    input_size=(224, 224),
    
    # Paths
    train_json_path="../../datasets/dataset_train.json",
    test_json_path="../../datasets/dataset_test.json",
    images_dir="../../datasets/images",
    save_dir="./checkpoints"
)
```

## Usage Examples

### Training from Python

```python
from baseline.config import BaselineConfig
from baseline.train import BaselineTrainer

# Create configuration
config = BaselineConfig(
    batch_size=64,
    max_epochs=100,
    learning_rate=1e-4
)

# Create and run trainer
trainer = BaselineTrainer(config)
trainer.train()

# Evaluate
results = trainer.evaluate()
print(f"Accuracy: {results['accuracy']:.4f}")

# Generate test predictions
submission_path = trainer.predict_test()
print(f"Submission saved to: {submission_path}")
```

### Command Line Training

```bash
# Basic training
python train.py

# Custom hyperparameters
python train.py --batch-size 64 --learning-rate 1e-3 --epochs 100

# Different model
python train.py --model resnet101 --batch-size 32

# Custom data paths
python train.py --train-json /path/to/train.json --images-dir /path/to/images

# Evaluation only
python train.py --evaluate-only --resume /path/to/checkpoint.pth

# Generate test predictions
python train.py --predict-test --resume /path/to/best_checkpoint.pth
```

### Running the Demo

```bash
# Show configuration examples
python demo.py --show-configs

# Show command line usage
python demo.py --show-cli

# Run training demo (requires dataset)
python demo.py --run-training
```

## Model Architecture

### Backbone
- **ResNet50** (default): Well-established CNN with proven FGVC performance
- **ResNet101**: Deeper variant for potentially better performance
- **EfficientNet-B3/B4**: More efficient alternatives

### Transfer Learning Strategy
1. **Freeze Phase** (first 5 epochs): Train only the classifier layer
2. **Fine-tuning Phase**: Unfreeze backbone with reduced learning rate
3. **Early Stopping**: Monitor validation accuracy with patience

### Input Processing
- **ROI Extraction**: Crop images to bounding box regions
- **Resize**: 224×224 pixels (ImageNet standard)
- **Normalization**: ImageNet statistics
- **Augmentation**: Random flips, rotations, color jittering (training only)

## Evaluation Metrics

### Standard Metrics
- **Top-1 Accuracy**: Primary development metric
- **Top-5 Accuracy**: Secondary metric
- **Per-class Metrics**: Precision, Recall, F1-score

### Competition Metric
- **Hierarchical Distance**: Using FathomNet WoRMS taxonomic tree
- Calculated via `fathomnet-py` integration
- Lower values are better (0 = perfect, 12 = maximum penalty)

## Output Files

Training produces the following outputs in the save directory:

```
checkpoints/
├── checkpoint_best.pth           # Best model weights
├── training.log                  # Training logs
├── training_metrics.json         # Metrics history
├── training_curves.png           # Loss/accuracy plots
├── validation_predictions.csv    # Validation predictions
├── test_predictions.csv          # Test predictions
├── submission.csv               # Competition submission file
├── evaluation_report.txt        # Detailed evaluation report
├── confusion_matrix.png         # Confusion matrix visualization
├── class_performance.png        # Per-class performance plots
└── logs/                        # TensorBoard logs
```

## Expected Performance

Based on the baseline plan, expected performance targets:

- **Top-1 Accuracy**: 40-60% (79-class fine-grained classification)
- **Hierarchical Distance**: 3-5 (baseline expectation)
- **Training Time**: 2-4 hours on single GPU

## Next Steps

After establishing the baseline, consider these improvements:

1. **Hierarchical Loss**: Modify loss function to incorporate taxonomic distances
2. **BioCLIP Fine-tuning**: Replace ImageNet with BioCLIP pre-trained weights
3. **Advanced Architectures**: Vision Transformers, attention mechanisms
4. **Data Enhancement**: Advanced augmentations, external data integration
5. **Ensemble Methods**: Combine multiple models

## Troubleshooting

### Common Issues

1. **Dataset Not Found**
   ```
   ERROR: Could not find dataset_train.json
   ```
   - Ensure you've downloaded the competition data
   - Update paths in config or use command line arguments

2. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   - Reduce batch size: `--batch-size 16`
   - Use gradient accumulation (modify training loop)

3. **Missing Dependencies**
   ```
   ImportError: No module named 'fathomnet'
   ```
   - Install missing packages: `pip install fathomnet-py`

4. **Low Performance**
   - Check data quality and preprocessing
   - Verify learning rate and batch size
   - Ensure proper validation split

### Performance Tips

1. **GPU Memory Optimization**
   - Use mixed precision training (`torch.cuda.amp`)
   - Reduce batch size and increase epochs proportionally
   - Use gradient checkpointing for larger models

2. **Training Speedup**
   - Increase `num_workers` for data loading
   - Use SSD storage for faster I/O
   - Pre-compute and cache ROI crops

3. **Model Selection**
   - Monitor both accuracy and hierarchical distance
   - Use validation set for hyperparameter tuning
   - Save multiple checkpoints for ensemble

## Contributing

To extend the baseline model:

1. **Add New Architectures**: Extend `model.py` with new backbone options
2. **Custom Loss Functions**: Modify `train.py` with hierarchical losses
3. **Advanced Augmentations**: Enhance `dataset.py` preprocessing
4. **Better Evaluation**: Extend `evaluate.py` with additional metrics

## References

- [FathomNet Competition](https://www.kaggle.com/competitions/fathomnet-fgvc-2025)
- [Baseline Model Plan](../../docs/PLAN_base_line_model.md)
- [Competition Strategy](../../docs/PLAN.md)
- [FathomNet API Documentation](https://fathomnet-py.readthedocs.io/)
- [BioCLIP Paper](https://imageomics.github.io/bioclip)