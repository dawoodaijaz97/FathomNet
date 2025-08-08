# Baseline Model Plan for FathomNet FGVC 2025

This document outlines a detailed plan for implementing the baseline model as the first step in the FathomNet FGVC 2025 competition approach.

## 1. Baseline Model Objectives

- **Primary Goal:** Establish a performance baseline using a simple, well-understood approach
- **Secondary Goals:** 
  - Validate data pipeline and evaluation metrics
  - Understand model performance without hierarchical considerations
  - Create a reference point for measuring improvements from advanced techniques
  - Identify potential issues early in the development process

## 2. Model Architecture

### 2.1 Pre-trained CNN Selection
- **Primary Choice:** ResNet50 pre-trained on ImageNet
  - Well-established architecture with proven FGVC performance
  - Good balance between model complexity and training speed
  - Strong feature extraction capabilities for fine-grained classification
- **Alternative Options:** 
  - EfficientNet-B3/B4 (if computational resources allow)
  - ResNet101 (for comparison with deeper network)

### 2.2 Model Configuration
- **Input:** ROI images (cropped regions of interest)
  - Rationale: ROIs provide focused view of target species, reducing background noise
  - Size: 224x224 pixels (standard ImageNet input size)
- **Output:** 79 classes (flat classification, ignoring taxonomic hierarchy initially)
- **Final Layer:** Replace pre-trained classifier with new fully connected layer
  - Input: Feature vector from pre-trained backbone
  - Output: 79-dimensional logits (one per category)

## 3. Data Pipeline

### 3.1 Data Preprocessing
1. **Image Loading:**
   - Extract ROI coordinates from COCO annotations
   - Crop full images to ROI bounding boxes
   - Handle edge cases (ROIs extending beyond image boundaries)

2. **Image Preprocessing:**
   - Resize ROIs to 224x224 pixels
   - Normalize using ImageNet statistics: 
     - Mean: [0.485, 0.456, 0.406]
     - Std: [0.229, 0.224, 0.225]

3. **Data Augmentation:**
   - **Training:** 
     - Random horizontal flip (probability: 0.5)
     - Random rotation (±15 degrees)
     - Random color jittering (brightness: 0.2, contrast: 0.2, saturation: 0.2, hue: 0.1)
     - Random crop with padding (maintain aspect ratio)
   - **Validation/Test:** Center crop only (no augmentation)

### 3.2 Dataset Splits
- **Training:** Use provided training annotations
- **Validation:** Create stratified 20% split from training data
  - Ensure each category has representation in validation set
  - Maintain class distribution as much as possible
- **Test:** Use provided test annotations for final evaluation

### 3.3 Data Loading
- **Batch Size:** 32 (adjust based on GPU memory)
- **Workers:** 4 parallel data loading processes
- **Shuffle:** True for training, False for validation/test

## 4. Training Strategy

### 4.1 Loss Function
- **Primary:** Standard Cross-Entropy Loss
  - Treats all misclassifications equally (ignores taxonomic hierarchy)
  - Simple and well-understood baseline

### 4.2 Optimization
- **Optimizer:** Adam with weight decay
  - Learning Rate: 1e-4 (start with pre-trained model learning rate)
  - Weight Decay: 1e-4
  - Betas: (0.9, 0.999)
- **Learning Rate Schedule:** 
  - ReduceLROnPlateau: Reduce LR by factor of 0.5 when validation loss plateaus
  - Patience: 5 epochs
  - Minimum LR: 1e-7

### 4.3 Training Configuration
- **Epochs:** 50 maximum (with early stopping)
- **Early Stopping:** 
  - Monitor validation accuracy
  - Patience: 10 epochs
  - Save best model based on validation performance
- **Fine-tuning Strategy:**
  - Freeze backbone weights for first 5 epochs (train only classifier)
  - Then unfreeze all layers and train end-to-end with lower learning rate

## 5. Evaluation Metrics

### 5.1 Standard Metrics (for development)
- **Primary:** Top-1 Accuracy
- **Secondary:** Top-5 Accuracy
- **Per-class:** Precision, Recall, F1-score for each of 79 categories

### 5.2 Competition Metric
- **Hierarchical Distance:** Implement using fathomnet-py WoRMS module
  - Calculate mean hierarchical distance between predictions and ground truth
  - Use this as the primary metric for model selection and comparison
  - Validate implementation against competition evaluation

## 6. Implementation Steps

### Phase 1: Data Pipeline Setup (Days 1-2)
1. Set up data loading from COCO annotations
2. Implement ROI extraction and preprocessing
3. Create validation split
4. Test data pipeline with small subset

### Phase 2: Model Implementation (Days 3-4)
1. Load pre-trained ResNet50
2. Modify final classification layer
3. Implement training loop with logging
4. Add model checkpointing and resuming capability

### Phase 3: Training and Evaluation (Days 5-7)
1. Train baseline model with cross-entropy loss
2. Monitor training progress and adjust hyperparameters
3. Implement hierarchical distance evaluation
4. Generate predictions on validation set

### Phase 4: Analysis and Documentation (Day 8)
1. Analyze model performance per category
2. Identify failure cases and potential improvements
3. Document results and lessons learned
4. Prepare baseline for comparison with advanced models

## 7. Expected Outcomes

### 7.1 Performance Targets
- **Top-1 Accuracy:** 40-60% (rough estimate for 79-class fine-grained classification)
- **Hierarchical Distance:** 3-5 (baseline expectation, lower is better)
- **Training Time:** 2-4 hours on single GPU

### 7.2 Key Insights Expected
- Understand which categories are most/least challenging
- Identify common misclassification patterns
- Assess impact of class imbalance on performance
- Validate data quality and preprocessing steps

## 8. Technical Implementation Details

### 8.1 Code Structure
```
src/baseline/
├── __init__.py
├── model.py           # Model definition and loading
├── dataset.py         # Dataset and data loading
├── train.py          # Training loop and logic
├── evaluate.py       # Evaluation metrics and testing
├── utils.py          # Helper functions
└── config.py         # Configuration parameters
```

### 8.2 Dependencies
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- Pillow >= 8.3.0
- fathomnet-py (for hierarchical distance)
- pycocotools (for COCO data handling)

### 8.3 Hardware Requirements
- **Minimum:** 1 GPU with 8GB VRAM
- **Recommended:** 1 GPU with 16GB VRAM
- **Storage:** ~50GB for downloaded images and model checkpoints

## 9. Success Criteria

### 9.1 Technical Success
- [ ] Model trains without errors
- [ ] Achieves reasonable accuracy on validation set
- [ ] Hierarchical distance evaluation works correctly
- [ ] Can generate predictions in competition format

### 9.2 Strategic Success
- [ ] Establishes clear baseline performance number
- [ ] Provides insights for advanced model development
- [ ] Validates competition evaluation pipeline
- [ ] Identifies key challenges in the dataset

## 10. Next Steps After Baseline

1. **Hierarchical Loss Integration:** Modify loss function to incorporate taxonomic distances
2. **BioCLIP Fine-tuning:** Replace ImageNet pre-training with BioCLIP weights
3. **Architecture Exploration:** Test more advanced architectures (Vision Transformers, etc.)
4. **Data Enhancement:** Experiment with additional augmentations and preprocessing techniques
5. **Ensemble Methods:** Combine multiple models for improved performance

This baseline model serves as the foundation for all subsequent improvements and provides essential insights into the competition dataset and evaluation methodology.
