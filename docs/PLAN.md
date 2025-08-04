# FathomNet FGVC 2025 Competition Plan

This document outlines a plan to approach the FathomNet FGVC 2025 Kaggle competition: "Navigating the Depths: Advancing Hierarchical Classification of Ocean Life".

## 1. Understanding the Task

*   **Goal:** Classify marine wildlife images into 79 categories of varying taxonomic ranks (kingdom, phylum, class, order, family, genus, species).
*   **Data:** COCO-formatted annotations (`dataset_train.json`, `dataset_test.json`) referencing images downloadable via URLs. Training set has 300 examples/category, test set ~10 examples/category. Includes full images and ROIs.
*   **Evaluation:** Mean Hierarchical Distance based on the taxonomic tree difference between the prediction and ground truth. Lower score is better (0=perfect, 12=worst/invalid). Uses FathomNet WoRMS module.
*   **Constraints:** Use only competition data and FathomNet Database images for training image data. Pre-trained models (BioCLIP, ImageNet, COCO) are allowed for initialization. Specify external data/models used.

## 2. Literature Review & Key Concepts

Focus on understanding techniques relevant to the competition:

*   **Hierarchical Classification:**
    *   General approaches (flat vs. hierarchical, local classifiers per level/node, global classifiers).
    *   Loss functions for hierarchical structures (e.g., hierarchical cross-entropy).
    *   Exploiting taxonomic relationships (e.g., graph neural networks, embedding hierarchies).
    *   **Key Resource:** [BioCLIP paper](https://imageomics.github.io/bioclip) (mentioned as a recent advance). Understand its architecture and how it incorporates biological knowledge.
*   **Fine-Grained Visual Categorization (FGVC):**
    *   Techniques for distinguishing subtle differences between classes (e.g., attention mechanisms, bilinear pooling, specialized architectures).
    *   Data augmentation strategies for FGVC.
*   **Transfer Learning & Pre-trained Models:**
    *   Effectiveness of different pre-trained weights (ImageNet, COCO, BioCLIP) for this specific domain (marine wildlife).
    *   Fine-tuning strategies vs. feature extraction.
*   **COCO Data Format:** Understand the structure of `dataset_train.json` and `dataset_test.json`.
*   **FathomNet WoRMS Module:** Review the [documentation](https://fathomnet-py.readthedocs.io/en/latest/api.html#module-fathomnet.api.worms) to understand how the taxonomic tree and distances are calculated for the evaluation metric.

## 3. Data Pipeline Strategy

*   **Download:**
    *   Use the provided `download.py` script.
    *   Ensure sufficient storage space.
    *   Adjust `--num-downloads` for optimal speed/stability.
    *   Handle potential download errors/retries.
    *   Store both full images and ROIs. Decide which to use for training/inference (ROIs seem more direct for classification).
*   **Data Loading & Parsing:**
    *   Use libraries like `pycocotools` or custom scripts to parse `dataset_train.json`.
    *   Create data loaders (e.g., PyTorch `Dataset` and `DataLoader`).
    *   Map category IDs to concept names and potentially fetch their full taxonomy using `fathomnet-py` early for model input or analysis.
*   **Preprocessing:**
    *   Standard image normalization (based on pre-trained model requirements).
    *   Resize images/ROIs consistently.
*   **Augmentation:**
    *   Standard augmentations: Random flips, rotations, color jittering.
    *   Consider FGVC-specific augmentations if applicable.
*   **Validation Split:**
    *   Create a stratified validation split from `dataset_train.json` to maintain class distribution.
    *   Ensure the split reflects the hierarchical nature if possible (e.g., don't put all species of one genus in the validation set).

## 4. Modeling Strategy

*   **Baseline Model:**
    *   Start simple: Fine-tune a standard pre-trained CNN (e.g., ResNet50, EfficientNet) on the ROIs using standard cross-entropy loss, ignoring the hierarchy initially. This provides a performance baseline.
*   **Incorporate Hierarchy:**
    *   **Option 1 (Hierarchical Loss):** Modify the loss function to penalize predictions based on taxonomic distance. Requires integrating the WoRMS distance calculation into the loss.
    *   **Option 2 (Hierarchical Architecture):** Explore models that explicitly model the hierarchy (e.g., Graph Neural Networks on the taxonomy, conditional prediction).
    *   **Option 3 (BioCLIP Fine-tuning):** Leverage BioCLIP's pre-training, which already incorporates biological concepts, and fine-tune it on the competition data. This seems promising given the competition's context.
*   **Pre-trained Weights:** Experiment with initializing models using ImageNet, COCO, and especially BioCLIP weights. Compare performance.
*   **Input Data:** Decide whether to use ROIs (likely sufficient) or full images (might provide more context but computationally expensive).

## 5. Evaluation Strategy

*   **Local Metric Implementation:**
    *   Install `fathomnet-py`.
    *   Implement a function that takes predictions (annotation ID, predicted concept name) and ground truth (`dataset_train.json` for validation) and calculates the mean hierarchical distance exactly as described in the competition rules.
    *   Requires the same taxonomic reference library used by the competition (likely accessible via `fathomnet-py`).
*   **Validation:** Use the local metric implementation on the validation set to monitor training progress and perform hyperparameter tuning.

## 6. Submission Strategy

*   **Output Format:** Generate predictions for the test set (`dataset_test.json`) annotations. Format the output as a CSV file with columns `annotation_id` and `concept_name`, matching `sample_submission.csv`.
*   **Concept Names:** Ensure predicted `concept_name` strings are valid taxonomic names present in the WoRMS reference library and belong to one of the 7 allowed ranks. Check for potential misspellings.
*   **Handling Uncertainty:** If the model is uncertain, consider predicting a higher taxonomic rank (e.g., family instead of species) to minimize potential distance penalties, rather than guessing a wrong species. Invalid names result in the maximum penalty (12).

## 7. Tools & Libraries

*   **Programming Language:** Python (>= 3.8)
*   **ML Framework:** PyTorch / TensorFlow / JAX
*   **Core Libraries:** `pandas`, `numpy`, `scikit-learn`, `PIL`/`Pillow`, `opencv-python`
*   **Data Handling:** `pycocotools`, `requests`, `aiohttp` (for downloader)
*   **Competition Specific:** `fathomnet-py` (essential for evaluation metric)
*   **Experiment Tracking:** MLflow / Weights & Biases (optional but recommended)

## 8. Milestones (Example)

1.  **Setup:** Environment, libraries, data download, basic data loading.
2.  **Baseline:** Train and evaluate a simple CNN baseline. Implement local evaluation metric.
3.  **Hierarchy Integration:** Implement and test a method incorporating hierarchy (e.g., BioCLIP fine-tuning or hierarchical loss).
4.  **Optimization:** Hyperparameter tuning, experiment with augmentations, architectures.
5.  **Submission:** Generate predictions for the test set, format, and submit. 