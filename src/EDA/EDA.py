#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pycocotools.coco import COCO
import fathomnet.api.worms as worms
import fathomnet.api as fathom
import os
import sys
# Add the parent directory (src) to the Python path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.worms_util import populate_worms_data



# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
sns.set_context("talk")

# File paths - try different possible locations
POSSIBLE_PATHS = [
    "../../datasets/dataset_train.json",  # Standard path from project root
    "../../dataset/dataset_train.json",   # Alternative path mentioned in README
]

def find_dataset_file(base_filename):
    """Find the correct path to the dataset file."""
    #print the current working directory
    print(f"Current working directory: {os.getcwd()}")
    for path in POSSIBLE_PATHS:
        test_path = path.replace("dataset_train.json", base_filename)
        if os.path.exists(test_path):
            print(f"Found dataset file at: {test_path}")
            return test_path
    
    # If we get here, we couldn't find the file
    print(f"ERROR: Could not find {base_filename} in any of these locations:")
    for path in POSSIBLE_PATHS:
        print(f"  - {path.replace('dataset_train.json', base_filename)}")
    
    # Ask for user input
    user_path = input(f"Please enter the full path to {base_filename}: ")
    if os.path.exists(user_path):
        return user_path
    else:
        raise FileNotFoundError(f"Could not find {base_filename} at {user_path}")

def load_datasets():
    """Load the COCO formatted datasets."""
    # Find dataset files
    train_json_path = find_dataset_file("dataset_train.json")
    test_json_path = find_dataset_file("dataset_test.json")
    
    print(f"Loading datasets from {train_json_path} and {test_json_path}")
    
    # Load datasets
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)
    
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    
    # Also load with pycocotools for easier access
    train_coco = COCO(train_json_path)
    test_coco = COCO(test_json_path)
    
    return train_data, test_data, train_coco, test_coco

def initial_inspection(train_data, test_data):
    """Examine the top-level structure of datasets."""
    print("\n=== Initial Dataset Inspection ===")
    
    # Check the top-level keys
    print(f"Training dataset keys: {list(train_data.keys())}")
    print(f"Test dataset keys: {list(test_data.keys())}")
    
    # Print basic counts
    print(f"\nTraining set:")
    print(f"  Number of images: {len(train_data['images'])}")
    print(f"  Number of annotations: {len(train_data['annotations'])}")
    print(f"  Number of categories: {len(train_data['categories'])}")
    
    print(f"\nTest set:")
    print(f"  Number of images: {len(test_data['images'])}")
    
    # Sample entries
    print("\nSample image entry:")
    print(json.dumps(train_data['images'][0], indent=2))
    
    print("\nSample annotation entry:")
    print(json.dumps(train_data['annotations'][0], indent=2))
    
    print("\nSample category entry:")
    print(json.dumps(train_data['categories'][0], indent=2))

def analyze_categories(train_data, train_coco):
    """Extract and analyze categories."""
    print("\n=== Category Analysis ===")
    
    # Extract all categories
    categories = {cat['id']: cat['name'] for cat in train_data['categories']}
    print(f"Number of categories: {len(categories)}")
    
    # Create a DataFrame for easier manipulation
    categories_df = pd.DataFrame(train_data['categories'])
    print("\nFirst 10 categories:")
    print(categories_df[['id', 'name']].head(10))
    
    # Count annotations per category
    cat_counts = Counter([ann['category_id'] for ann in train_data['annotations']])
    cat_count_df = pd.DataFrame({
        'category_id': list(cat_counts.keys()),
        'count': list(cat_counts.values())
    })
    
    # Merge with category names
    cat_count_df = cat_count_df.merge(
        categories_df[['id', 'name']], 
        left_on='category_id', 
        right_on='id'
    )
    
    # Sort by count
    cat_count_df = cat_count_df.sort_values('count', ascending=False)
    
    print("\nAnnotation counts per category (top 10):")
    print(cat_count_df[['name', 'count']].head(10))
    
    # Plot annotation frequency
    plt.figure(figsize=(15, 8))
    plt.bar(range(len(cat_count_df)), cat_count_df['count'])
    plt.title('Number of Annotations per Category')
    plt.xlabel('Category Index (sorted by frequency)')
    plt.ylabel('Annotation Count')
    plt.tight_layout()
    plt.savefig('category_annotation_counts.png')
    
    return categories_df, cat_count_df

def retrieve_taxonomic_info(categories_df):
    print(categories_df)
    """Retrieve taxonomic information using fathomnet-py."""
    print("\n=== Taxonomic Hierarchy Analysis ===")
    
    ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    TAXONOMIC_RANKS = [
        "Domain",
        "Kingdom",
        "Phylum",
        "Subphylum",
        "Gigaclass",   #optionalrare
        "Class",
        "Order",
        "Infraorder",
        "Family",
        "Subfamily",
        "Genus",
        "Species"
    ]
    
    #create a with columns 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'
    ranks_list = []
    for _, category in categories_df.iterrows():
        print(category["name"]) 
        info = worms.get_info(category["name"])
        #get all the ranks and create a distribution of the ranks
        ranks_list.append(info.rank)
        print(ranks_list)
        #create a distribution of the ranks
        rank_distribution = Counter(ranks_list)
        print(rank_distribution)
        #plot the distribution
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(rank_distribution.keys()), y=list(rank_distribution.values()))
        plt.title('Distribution of Ranks')
        plt.xlabel('Rank')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('rank_distribution.png')
        #save the rank distribution to a csv file
    



    
   

def analyze_annotations(train_data):
    """Analyze annotation properties."""
    print("\n=== Annotation Analysis ===")
    
    # Create DataFrame of annotations
    annotations_df = pd.DataFrame(train_data['annotations'])
    
    # Count annotations per image
    ann_per_img = annotations_df['image_id'].value_counts().reset_index()
    ann_per_img.columns = ['image_id', 'annotation_count']
    
    print(f"\nAnnotations per image statistics:")
    print(f"  Mean: {ann_per_img['annotation_count'].mean():.2f}")
    print(f"  Median: {ann_per_img['annotation_count'].median():.2f}")
    print(f"  Min: {ann_per_img['annotation_count'].min()}")
    print(f"  Max: {ann_per_img['annotation_count'].max()}")
    
    # Plot annotations per image distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(ann_per_img['annotation_count'], bins=20)
    plt.title('Distribution of Annotations per Image')
    plt.xlabel('Number of Annotations')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('annotations_per_image.png')
    
    # Analyze ROI (bounding box) properties
    # Extract bounding box coordinates
    annotations_df['x'] = annotations_df['bbox'].apply(lambda x: x[0])
    annotations_df['y'] = annotations_df['bbox'].apply(lambda x: x[1])
    annotations_df['width'] = annotations_df['bbox'].apply(lambda x: x[2])
    annotations_df['height'] = annotations_df['bbox'].apply(lambda x: x[3])
    
    # Calculate area and aspect ratio
    annotations_df['area'] = annotations_df['width'] * annotations_df['height']
    annotations_df['aspect_ratio'] = annotations_df['width'] / annotations_df['height']
    
    print(f"\nROI area statistics:")
    print(f"  Mean: {annotations_df['area'].mean():.2f}")
    print(f"  Median: {annotations_df['area'].median():.2f}")
    print(f"  Min: {annotations_df['area'].min():.2f}")
    print(f"  Max: {annotations_df['area'].max():.2f}")
    
    print(f"\nROI aspect ratio statistics:")
    print(f"  Mean: {annotations_df['aspect_ratio'].mean():.2f}")
    print(f"  Median: {annotations_df['aspect_ratio'].median():.2f}")
    print(f"  Min: {annotations_df['aspect_ratio'].min():.2f}")
    print(f"  Max: {annotations_df['aspect_ratio'].max():.2f}")
    
    # Plot ROI area distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(annotations_df['area'], bins=50, log_scale=True)
    plt.title('Distribution of ROI Areas (log scale)')
    plt.xlabel('Area (pixels²)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('roi_area_distribution.png')
    
    # Plot ROI aspect ratio distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(annotations_df['aspect_ratio'], bins=50)
    plt.title('Distribution of ROI Aspect Ratios')
    plt.xlabel('Aspect Ratio (width/height)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('roi_aspect_ratio_distribution.png')
    
    return annotations_df

def run_eda():
    """Run the full EDA process."""
    # Load datasets
    train_data, test_data, train_coco, test_coco = load_datasets()
    
    # Initial dataset inspection
    initial_inspection(train_data, test_data)
    
    # Analyze categories
    categories_df, cat_count_df = analyze_categories(train_data, train_coco)
    
    # Retrieve and analyze taxonomic information
    taxonomy_df = retrieve_taxonomic_info(categories_df)
    
    # Analyze annotations
    ##annotations_df = analyze_annotations(train_data)
    
    print("\nEDA completed! Visualizations saved to current directory.")
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'train_coco': train_coco,
        'test_coco': test_coco,
        'categories_df': categories_df,
        'cat_count_df': cat_count_df,
        'taxonomy_df': taxonomy_df,
        'annotations_df': annotations_df
    }

if __name__ == "__main__":
    results = run_eda()


