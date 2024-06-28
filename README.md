# Leaf Classification and Clustering

This project involves classifying and clustering different types of leaves based on their features and texture. The project uses several machine learning techniques to achieve classification and clustering, and visualizes the results using t-SNE.

## Table of Contents
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Results](#results)
- [Conclusion](#conclusion)

## Dataset

The dataset used in this project consists of leaf images and their corresponding features. The features are loaded from a CSV file, and the images are loaded from a specified directory.

- CSV file: `leaves.csv`
- Images directory: `leaves`

The CSV file contains various features extracted from the leaf images. Each row corresponds to a leaf image and its features.

## Requirements

The project requires the following Python packages:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- opencv-python
- scikit-image
- scipy

You can install the required packages using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn opencv-python scikit-image scipy
```

## Usage

1. Clone the repository or download the source code.
2. Place the `leaves.csv` file and the `leaves` directory in the appropriate locations.
3. Run the Jupyter notebook or Python script.

## Code Overview

### 1. Data Loading and Preparation

- Load the CSV file containing leaf features.
- Map class labels to their corresponding names.
- Load the leaf images from the specified directory.

### 2. Feature Extraction and Dataset Augmentation

- Extract texture features from the images using Grey Level Co-occurrence Matrix (GLCM).
- Combine the original features with the extracted texture features.

### 3. Classification

- Split the data into training and test sets.
- Train a classification pipeline using feature selection and an ExtraTreesClassifier.
- Evaluate the classification performance using accuracy, classification report, and confusion matrix.

### 4. Clustering

- Standardize the features and apply Principal Component Analysis (PCA) for dimensionality reduction.
- Perform agglomerative clustering and evaluate its performance using several metrics (silhouette score, homogeneity, completeness, V-measure, ARI, NMI).

### 5. Visualization

- Use t-SNE to visualize the clusters and true classes in 2D space.

## Results

### Classification

- **Accuracy**: 100%
- **Confusion Matrix**: The confusion matrix indicates perfect classification.

### Clustering

- **Silhouette Score**: 0.28
- **Homogeneity**: 0.74
- **Completeness**: 0.77
- **V-measure**: 0.75
- **Adjusted Rand Index (ARI)**: 0.42
- **Normalized Mutual Information (NMI)**: 0.75

### Visualization

- t-SNE plots provide a visual representation of the clusters and true classes, showing how well the clustering algorithm performed.
