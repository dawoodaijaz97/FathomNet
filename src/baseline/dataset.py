"""
Dataset implementation for FathomNet baseline model.
"""

import json
import os
import random
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from .config import BaselineConfig


class FathomNetDataset(Dataset):
    """
    PyTorch Dataset for FathomNet competition data.
    
    Loads images and ROI annotations from COCO format data.
    """
    
    def __init__(
        self,
        annotations: List[Dict],
        images_info: Dict[int, Dict],
        categories_info: Dict[int, Dict],
        images_dir: str,
        transform: Optional[transforms.Compose] = None,
        use_roi: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            annotations: List of annotation dictionaries from COCO format
            images_info: Dictionary mapping image_id to image information
            categories_info: Dictionary mapping category_id to category information
            images_dir: Directory containing the images
            transform: Optional transforms to apply to images
            use_roi: Whether to crop images to ROI bounding boxes
        """
        self.annotations = annotations
        self.images_info = images_info
        self.categories_info = categories_info
        self.images_dir = images_dir
        self.transform = transform
        self.use_roi = use_roi
        
        # Create mapping from category names to indices
        self.category_names = sorted([cat['name'] for cat in categories_info.values()])
        self.name_to_idx = {name: idx for idx, name in enumerate(self.category_names)}
        self.idx_to_name = {idx: name for name, idx in self.name_to_idx.items()}
        
    def __len__(self) -> int:
        """Return the number of annotations in the dataset."""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (image_tensor, label, metadata)
        """
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        
        # Get image info
        image_info = self.images_info[image_id]
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
        
        # Extract ROI if specified
        if self.use_roi and 'bbox' in annotation:
            bbox = annotation['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # Ensure coordinates are within image bounds
            img_width, img_height = image.size
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            
            # Crop to ROI
            image = image.crop((x, y, x + w, y + h))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        category_name = self.categories_info[category_id]['name']
        label = self.name_to_idx[category_name]
        
        # Metadata
        metadata = {
            'annotation_id': annotation['id'],
            'image_id': image_id,
            'category_id': category_id,
            'category_name': category_name,
            'image_path': image_path
        }
        
        return image, label, metadata


def load_coco_data(json_path: str) -> Tuple[List[Dict], Dict[int, Dict], Dict[int, Dict]]:
    """
    Load COCO format data from JSON file.
    
    Args:
        json_path: Path to the COCO JSON file
        
    Returns:
        Tuple of (annotations, images_info, categories_info)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    
    # Create lookup dictionaries
    images_info = {img['id']: img for img in data['images']}
    categories_info = {cat['id']: cat for cat in data['categories']}
    
    return annotations, images_info, categories_info


def create_validation_split(
    annotations: List[Dict],
    validation_split: float = 0.2,
    stratify: bool = True,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create train/validation split from annotations.
    
    Args:
        annotations: List of annotation dictionaries
        validation_split: Fraction of data to use for validation
        stratify: Whether to maintain class distribution in split
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_annotations, val_annotations)
    """
    if stratify:
        # Extract labels for stratification
        labels = [ann['category_id'] for ann in annotations]
        
        # Create stratified split
        train_indices, val_indices = train_test_split(
            range(len(annotations)),
            test_size=validation_split,
            stratify=labels,
            random_state=random_seed
        )
    else:
        # Create random split
        random.seed(random_seed)
        indices = list(range(len(annotations)))
        random.shuffle(indices)
        
        val_size = int(len(indices) * validation_split)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
    
    train_annotations = [annotations[i] for i in train_indices]
    val_annotations = [annotations[i] for i in val_indices]
    
    return train_annotations, val_annotations


def get_transforms(config: BaselineConfig, is_training: bool = True) -> transforms.Compose:
    """
    Get image transforms for training or validation.
    
    Args:
        config: Configuration object
        is_training: Whether this is for training (enables augmentation)
        
    Returns:
        Composed transforms
    """
    if is_training:
        # Training transforms with augmentation
        transform_list = [
            transforms.Resize((config.input_size[0], config.input_size[1])),
            transforms.RandomHorizontalFlip(p=config.horizontal_flip_prob),
            transforms.RandomRotation(degrees=config.rotation_degrees),
            transforms.ColorJitter(
                brightness=config.color_jitter_brightness,
                contrast=config.color_jitter_contrast,
                saturation=config.color_jitter_saturation,
                hue=config.color_jitter_hue
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.normalize_mean,
                std=config.normalize_std
            )
        ]
    else:
        # Validation/test transforms without augmentation
        transform_list = [
            transforms.Resize((config.input_size[0], config.input_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.normalize_mean,
                std=config.normalize_std
            )
        ]
    
    return transforms.Compose(transform_list)


def create_data_loaders(config: BaselineConfig) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and validation data loaders.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader, dataset_info)
    """
    # Load training data
    print(f"Loading training data from {config.train_json_path}")
    train_annotations, images_info, categories_info = load_coco_data(config.train_json_path)
    
    print(f"Loaded {len(train_annotations)} annotations for {len(categories_info)} categories")
    
    # Create train/validation split
    print(f"Creating {config.validation_split:.1%} validation split")
    train_anns, val_anns = create_validation_split(
        train_annotations,
        validation_split=config.validation_split,
        stratify=config.stratify,
        random_seed=config.random_seed
    )
    
    print(f"Train: {len(train_anns)} annotations, Validation: {len(val_anns)} annotations")
    
    # Get transforms
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)
    
    # Create datasets
    train_dataset = FathomNetDataset(
        annotations=train_anns,
        images_info=images_info,
        categories_info=categories_info,
        images_dir=config.images_dir,
        transform=train_transform,
        use_roi=True
    )
    
    val_dataset = FathomNetDataset(
        annotations=val_anns,
        images_info=images_info,
        categories_info=categories_info,
        images_dir=config.images_dir,
        transform=val_transform,
        use_roi=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    # Dataset info for reference
    dataset_info = {
        'num_classes': len(categories_info),
        'class_names': train_dataset.category_names,
        'name_to_idx': train_dataset.name_to_idx,
        'idx_to_name': train_dataset.idx_to_name,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset)
    }
    
    return train_loader, val_loader, dataset_info


def create_test_loader(config: BaselineConfig, dataset_info: Dict) -> DataLoader:
    """
    Create test data loader.
    
    Args:
        config: Configuration object
        dataset_info: Dataset information from training
        
    Returns:
        Test data loader
    """
    # Load test data
    print(f"Loading test data from {config.test_json_path}")
    test_annotations, test_images_info, _ = load_coco_data(config.test_json_path)
    
    print(f"Loaded {len(test_annotations)} test annotations")
    
    # Get test transforms
    test_transform = get_transforms(config, is_training=False)
    
    # Create test dataset
    test_dataset = FathomNetDataset(
        annotations=test_annotations,
        images_info=test_images_info,
        categories_info=dataset_info['name_to_idx'],  # Use same category mapping
        images_dir=config.images_dir,
        transform=test_transform,
        use_roi=True
    )
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return test_loader