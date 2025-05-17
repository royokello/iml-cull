# iml-cull
A simple image culling toolkit with manual labeling built on Flask, AI-assisted culling recommendations based on Google ViT, and automated image management.

## Overview

iml-cull is a toolkit for image culling, which helps photographers and image editors efficiently manage large collections of photos by identifying and separating higher-quality images. The toolkit consists of three primary modules:

1. **Label Module** - Manual image labeling interface
2. **Train Module** - Training a custom Vision Transformer (ViT) model
3. **Cull Module** - Automated image selection based on the trained model

## Project Structure

iml-cull now supports a unified model approach with multiple albums. Create a directory structure as follows:

```
root_directory/             # Main root directory
├── album1/                 # First album folder
│   ├── src/                # Source images for album1
│   ├── src_culled/         # Directory for kept images (created automatically)
│   └── cull_labels.csv     # Labels file for this album (created during labeling)
├── album2/                 # Second album folder
│   ├── src/                # Source images for album2
│   ├── src_culled/         # Directory for kept images (created automatically)
│   └── cull_labels.csv     # Labels file for this album (created during labeling)
└── cull_model.pth          # Single trained model (created during training)
```

## Installation Requirements

This toolkit requires the following Python libraries:

```
Flask
torch
transformers
Pillow
```

You can install these dependencies using pip:

```
pip install flask torch transformers Pillow
```

## Module Usage

### 1. Label Module (`label.py`)

The label module provides a simple web interface to manually categorize images as "keep" or "cull".

**Usage:**
```
python label.py <album_directory>
```

Where `<album_directory>` is the path to your album folder containing the `src` directory with your images.

**Features:**
- Simple interface with "Keep" and "Cull" buttons
- Images displayed at a maximum height of 384px
- Navigation controls: Previous, Next, and Random buttons
- Statistics dashboard showing Keep, Cull, Unlabeled, and Total counts
- Support for removing labels
- Selective labeling capability
- Automatically saves labels to `cull_labels.csv`

**Workflow:**
1. Start the application and navigate to `http://127.0.0.1:5000` in your browser
2. Label images as "Keep" or "Cull" as needed
3. Navigate through images using Prev, Next, or Random buttons
4. Use the "Remove Label" button to clear a label if needed
5. All labels are automatically saved to `cull_labels.csv`

### 2. Train Module (`train.py`)

The train module now supports training a single model using data from multiple albums. It combines all labeled data from various albums to create a unified model.

**Usage:**
```
python train.py <root_directory>
```

Where `<root_directory>` is the path containing multiple album folders, each with their own labeled data.

**Optional Arguments:**
```
--epochs N           Specify number of training epochs (default: 256)
--batch-size N       Specify batch size (default: 64)
--learning-rate N    Specify learning rate (default: 1e-4)
--patience N         Specify early stopping patience (default: 16)
```

**Example:**
```
python train.py my_photos_root --epochs 100 --learning-rate 1e-5
```

**Features:**
- Uses Google's ViT (Vision Transformer) architecture
- Automatically combines data from multiple albums
- Automatically splits data into training and validation sets
- Implements early stopping to prevent overfitting
- Saves the best model based on validation performance
- Supports both CPU and GPU training (automatically detects CUDA)
- Default batch size of 64
- Default learning rate of 1e-4
- Default patience of 16 epochs for early stopping
- 25% validation split

The trained model is saved to `<root_directory>/cull_model.pth` and can be used for all albums within the root directory.

### 3. Cull Module (`cull.py`)

The cull module now supports processing multiple albums using a unified model. It automatically identifies and copies images worth keeping to each album's `src_culled` directory.

**Usage:**
```
python cull.py <root_directory>
```

Where `<root_directory>` is the path containing multiple album folders.

**Optional Arguments:**
```
--album NAME    Process only a specific album in the root directory
```

**Example:**
```
python cull.py my_photos_root             # Process all albums
python cull.py my_photos_root --album vacation  # Process only the vacation album
```

**Features:**
- Loads a single model for all albums
- Processes all albums in the root directory
- For each album, processes all images in the `src` directory
- Copies images predicted as "keep" to the `src_culled` directory
- Leaves original files untouched
- Provides detailed statistics for each album
- Shows a comprehensive summary of all processed albums

**Terminology:**
- "Kept" images are those copied to the `src_culled` directory (model predicts they should be kept)
- "Culled" images are those not copied (model predicts they should be removed)

**Workflow:**
1. Train a unified model using `train.py`
2. Run `cull.py` to automatically process all albums
3. Review the kept images in each album's `src_culled` directory
4. Manually verify the model's decisions

## Complete Workflow Example

```
# Step 1: Install dependencies
pip install flask torch transformers Pillow

# Step 2: Create your multi-album structure
mkdir -p my_photos/wedding/src my_photos/vacation/src my_photos/portrait/src
# Copy images to each album's src folder

# Step 3: Manual labeling (repeat for each album)
python label.py my_photos/wedding
python label.py my_photos/vacation
python label.py my_photos/portrait

# Step 4: Train a unified model
python train.py my_photos

# Step 5: Automatic culling of all albums
python cull.py my_photos
```

After these steps, each album will have a `src_culled` folder containing the images the model predicted should be kept.

## Notes

- The training process combines data from multiple albums for better generalization
- A single model is used across all albums for consistent results
- GPU acceleration is recommended for faster training but not required
- The quality of predictions depends on the consistency of your manual labeling
- The default learning rate (1e-4) is optimized for transfer learning with ViT

## CNN Module

The CNN module provides deep learning functionality for image classification using PyTorch. It uses a pre-trained ResNet-18 model fine-tuned for binary classification of images.

### Model Architecture
- Base model: ResNet-18 (pretrained on ImageNet)
- Modified for binary classification (2 output classes)
- Input size: 224x224 RGB images
- Output: Binary classification (cull/keep)

### Dataset
The `IMLCullDataset` class handles the image dataset:
- Reads image paths and labels from a CSV file
- Automatically resizes images to 224x224
- Applies standard ImageNet normalization
- Labels are mapped as: "cull": 0, "keep": 1

### Training Script
The training script (`cnn/train.py`) provides the following features:

```bash
python -m cnn.train --input <data_dir> [options]
```

Options:
- `--input`: Root directory containing the source folder with images (required)
- `--source`: Name of the source directory containing images (default: "src")
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-4)
- `--patience`: Number of epochs to wait before early stopping (default: 3)

Features:
- Uses Adam optimizer
- Cross-entropy loss for classification
- Early stopping based on validation accuracy
- Saves best model weights to `cull_model.pth`
- Saves training metadata to `cull_model.json`
- Automatic train/validation split (80/20)
- GPU support when available

### Model Output
The training process saves two files:
1. `cull_model.pth`: The best model weights based on validation accuracy
2. `cull_model.json`: Training metadata including:
   - Base model name
   - Number of epochs trained
   - Final validation accuracy
   - Learning rate
   - Batch size
