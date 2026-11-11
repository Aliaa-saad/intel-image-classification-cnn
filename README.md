# Intel Image Classification Using CNN

## Project Overview

This project aims to classify natural scene images into six categories using a Convolutional Neural Network (CNN) built and trained from scratch.

The image preprocessing and data augmentation pipelines are implemented using Python and TensorFlow/Keras.

## Dataset

**Name:** Intel Image Classification

**Source:** https://www.kaggle.com/datasets/puneet6060/intel-image-classification

The dataset contains:

- Training dataset: 14,034 images
- Test dataset: 3,000 images
- Number of classes: 6

**Classes:** Buildings, Forest, Glacier, Mountain, Sea, Street.

## Project Structure

- `src/explore.py`: Dataset exploration and image validation
- `src/preprocessing.py`: Dataset loading, resizing, normalization, and splitting
- `src/augmentation.py`: Random image augmentation
- `src/check_pipeline.py`: Input pipeline validation and sample visualization
- `data/`: Local dataset directory (not uploaded to GitHub)
- `results/`: Generated outputs

## Preprocessing

The preprocessing pipeline includes:

1. Dataset exploration and image validation
2. Resizing images to 128 × 128
3. RGB image conversion
4. Normalization to the range [0, 1]
5. Splitting the original training dataset into 80% training and 20% validation
6. Keeping the original test dataset separate

## Data Augmentation

Data augmentation is applied only to the training data:

- Random horizontal flipping
- Random rotation
- Random zooming

These operations are intended to improve generalization and reduce overfitting.

## Setup

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the Intel Image Classification dataset from Kaggle and extract it into the `data/` directory.

Expected paths:

```text
data/seg_train/seg_train/
data/seg_test/seg_test/
```

## Running the Preprocessing Pipeline

From the project root:

```bash
python -m src.explore
python -m src.check_pipeline
```

## Using the Prepared Data

The CNN training module can import:

```python
from src.preprocessing import load_datasets

train_ds, val_ds, test_ds, class_names = load_datasets()
```

The returned images are normalized to [0, 1], resized to 128 × 128, and batched. Training images are augmented by default.

## Model Training

CNN architecture, training, evaluation, and prediction components will be integrated after the preprocessing pipeline is validated.

## Overfitting Prevention

The pipeline uses data augmentation and a separate validation split. Additional techniques, such as Dropout and Early Stopping, should be implemented during CNN model training.

## Dataset Cleaning

The original Intel Image Classification dataset contained
exact duplicate images across training and testing data,
including some images with conflicting class labels.

To prevent exact-duplicate data leakage, we implemented
a cleaning script that:

- Detects duplicate files using SHA256 hashing.
- Excludes images with conflicting class labels.
- Removes redundant copies.
- Preserves the original dataset.
- Generates a cleaning report.

### Cleaning Results

Original dataset: 17,034 images
Removed files: 31
Clean dataset: 17,003 images

Clean training: 14,006 images
Clean testing: 2,997 images

After splitting the clean training set:

- Training: 11,205 images
- Validation: 2,801 images
- Testing: 2,997 images

### Required Setup

After downloading and extracting the dataset, run:

```bash
python -m src.clean_dataset
python -m src.explore
python -m src.check_pipeline
python -m src.quality_checks
```

The cleaning step must be completed before CNN training.

Exact duplicate hashes across the final splits: 0.
