# FathomNet FGVC 2025 Competition: EDA Plan

This document outlines the plan for Exploratory Data Analysis (EDA) on the FathomNet FGVC 2025 competition dataset.

## 1. Goals

*   Understand the structure and format of the COCO annotations (`dataset_train.json`, `dataset_test.json`).
*   Analyze the distribution of images, annotations, and categories.
*   Investigate the taxonomic hierarchy of the categories using `fathomnet-py`.
*   Examine image characteristics (size, quality) and ROI properties (size, aspect ratio).
*   Identify potential data issues like class imbalance, low-quality images, or annotation inconsistencies.
*   Visualize data distributions and relationships to inform preprocessing and modeling strategies.
*   Compare basic properties of the training and test sets.

## 2. Data Loading and Initial Inspection

*   Load `dataset_train.json` using `pycocotools` or a similar library.
*   Load `dataset_test.json`.
*   Examine the top-level structure: `info`, `licenses`, `images`, `annotations`, `categories`.
*   Print the number of images, annotations, and categories in the training set.
*   Print the number of images in the test set.
*   Inspect a few sample image entries, annotation entries, and category entries.

## 3. Category Analysis

*   **List Categories:** Extract all 79 category names and IDs.
*   **Taxonomic Hierarchy:**
    *   Use `fathomnet-py` (WoRMS module) to retrieve the full taxonomic lineage (kingdom, phylum, class, order, family, genus, species) for each of the 79 `concept_name`s.
    *   Store this taxonomic information, potentially mapping category IDs to their full lineage and rank.
    *   Visualize the taxonomic tree structure connecting the 79 categories. Identify shared parent nodes.
*   **Rank Distribution:**
    *   Determine the taxonomic rank for each of the 79 ground truth categories provided.
    *   Plot the distribution of these ranks (how many species, genera, families, etc., are target categories?).
*   **Annotation Frequency:**
    *   Calculate the number of training annotations for each category.
    *   Plot the distribution of annotations per category (histogram/bar chart). Identify highly frequent and rare categories (potential imbalance).

## 4. Annotation Analysis (Training Set)

*   **Annotations per Image:** Calculate and plot the distribution of the number of annotations per image.
*   **ROI Analysis:**
    *   Extract bounding box (`bbox`) information: `[x, y, width, height]`.
    *   Calculate ROI area (`width * height`) and aspect ratio (`width / height`).
    *   Plot distributions of ROI area and aspect ratio. Are there many very small or oddly shaped ROIs?
    *   Visualize some example ROIs overlaid on their images, especially extreme cases (very small/large areas or aspect ratios).
*   **Overlapping ROIs:** Check if any images have overlapping ROIs. If so, investigate the categories associated with them (are they related taxonomically?).

## 5. Image Analysis

*   **Image Dimensions:**
    *   Extract `width` and `height` for all training and test images.
    *   Plot distributions of image width, height, and aspect ratio for both train and test sets. Compare the distributions.
*   **Visual Inspection:**
    *   Display random sample images from the training set.
    *   Display random sample ROIs from different categories.
    *   Look for variations in image quality, lighting, background complexity, and viewpoint.
    *   Specifically look at examples from visually similar categories, especially those close in the taxonomic tree but different species/genera.
*   **(Optional) Color Analysis:** Analyze color distributions if relevant (e.g., histogram of pixel intensities).

## 6. Relationship Analysis

*   Relate ROI size/aspect ratio to specific categories or taxonomic ranks. Are certain types of animals typically captured in smaller/larger ROIs?
*   Analyze if image dimensions correlate with the number or type of annotations present.

## 7. Tools

*   **Programming Language:** Python
*   **Core Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `PIL`/`Pillow`, `opencv-python`
*   **Data Handling:** `pycocotools`
*   **Competition Specific:** `fathomnet-py` (essential for taxonomy)

## 8. Deliverables

*   Jupyter Notebook or Python script containing the EDA code.
*   Summary of findings, including key statistics, visualizations, identified data issues, and implications for modeling (e.g., handling imbalance, required augmentations, relevant image/ROI sizes). 