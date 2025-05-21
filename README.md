# IML Cull

IML Cull is a Python library for intelligent machine learning-based image culling that uses a Vision Transformer (ViT) model to learn from labeled samples and automatically select which images to keep or discard from a collection.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Modules](#modules)
  - [Label Module](#label-module)
  - [Train Module](#train-module)
  - [Cull Module](#cull-module)

## Overview

IML Cull helps photographers and image editors efficiently manage large collections of photos by identifying and separating higher-quality images. The toolkit consists of three primary modules:

1. **Label Module** - Manual image labeling interface
2. **Train Module** - Training a custom Vision Transformer (ViT) model
3. **Cull Module** - Automated image selection based on the trained model

## Project Structure

The project uses a stage-based workflow structure:

```
project_directory/
├── stage_1/            # First stage images directory
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── stage_1_cull_labels.csv  # Labels for stage 1 images
├── stage_1_cull_model.pth   # Trained model for stage 1
├── stage_1_cull_epoch_log.csv  # Training progress log
├── stage_2/            # Second stage (output from stage 1 culling)
│   ├── image1.jpg
│   └── ...
└── ...
```

## Installation

```bash
git clone https://github.com/royokello/iml-cull.git
cd iml-cull
pip install -r requirements.txt
```

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

## Modules

### Label Module

The label module provides a web-based interface for manually labeling images as "keep" or "cull".

#### Usage

```bash
python label.py --project "path/to/project" [--stage STAGE_NUMBER]
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|--------|
| `--project` | Root project directory containing images | Required |
| `--stage` | Stage number to label (e.g., 1, 2, etc.) | Latest stage number |

**Note:** At least one stage directory (e.g., 'stage_1') must exist in the project directory. The module will raise an error if no stage directories are found.

#### Features

- Interactive web interface (runs on port 5000)
- Simple "Keep" and "Cull" buttons for each image
- Navigation between images (previous, next, random)
- Auto-saving of labels to CSV file (`stage_{stage}_cull_labels.csv`)
- Statistics display showing distribution of labeled images

#### CSV Format

The module generates a CSV file with the following columns:
- `image_name`: Image filename
- `label`: String value representing the label ("keep" or "cull")

#### How It Works

1. The label interface loads images from the specified stage directory
2. Users view each image and decide whether to keep or cull it
3. Labels are automatically saved to a CSV file as you navigate between images
4. The system tracks statistics on how many images are labeled in each category

### Train Module

The train module trains a Vision Transformer model to predict whether images should be kept or culled based on labeled samples.

#### Usage

```bash
python train.py --project "path/to/project" [--stage STAGE_NUMBER] [options]
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|--------|
| `--project` | Root project directory | Required |
| `--stage` | Stage number to train (e.g., 1, 2, etc.) | Latest stage number |
| `--epochs` | Number of training epochs | 256 |
| `--batch-size` | Batch size | 64 |
| `--learning-rate` | Learning rate | 1e-4 |
| `--patience` | Early stopping patience in epochs | 8 |

**Note:** At least one stage directory (e.g., 'stage_1') must exist in the project directory. The module will raise an error if no stage directories are found.

#### Features

- Binary classification model architecture (keep/cull)
- CSV logging with per-epoch metrics for both training and validation
- Model checkpoint saving based on improvement in validation accuracy
- Early stopping to prevent overfitting
- Uses Google's ViT (Vision Transformer) architecture
- Automatically splits data into training and validation sets (25% validation split)
- Supports both CPU and GPU training (automatically detects CUDA)

#### Training Process

1. Reads labels from `stage_{stage}_cull_labels.csv` file in the specified stage directory
2. Splits data into training and validation sets based on the specified ratio
3. Trains the ViT model with a binary classification head
4. Logs progress to CSV file with columns:
   - epoch
   - train_loss
   - val_loss
   - val_accuracy
5. Saves best model as `stage_{stage}_cull_model.pth` when validation accuracy improves
6. Uses early stopping when no improvement is seen after the specified patience epochs

### Cull Module

The cull module uses a trained model to automatically process and select which images to keep from a specified stage directory.

#### Usage

```bash
python cull.py --project "path/to/project" [--stage STAGE_NUMBER]
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|--------|
| `--project` | Root project directory | Required |
| `--stage` | Stage number to process | Latest stage number |

#### Features

- Automatic batch processing of all images in a stage directory
- Dynamic output directory naming based on stage progression
- Detailed per-image logging of keep/cull decisions

#### Culling Process

1. Loads trained model from `stage_{stage}_cull_model.pth`
2. For each image in the stage directory:
   - Predicts whether the image should be kept (0) or culled (1)
   - If the prediction is "keep" (0), copies the image to the next stage directory
   - If the prediction is "cull" (1), the image is not copied
3. Creates a new stage directory (`stage_{stage+1}`) containing only the kept images

**Important Note:** In this project, "culling" refers to the process of selecting images to **remove**, while "keeping" refers to images that should be **preserved**. The next stage folder will contain images predicted as "keep" (prediction=0), not the ones that should be removed (prediction=1).

## Workflow Example

1. **Label**: Label images in stage_1 directory
   ```bash
   python label.py --project "project_dir" --stage 1
   ```

2. **Train**: Train a model using the labeled data
   ```bash
   python train.py --project "project_dir" --stage 1 --epochs 50
   ```

3. **Cull**: Apply the trained model to cull images
   ```bash
   python cull.py --project "project_dir" --stage 1
   ```
   This will create a stage_2 directory with only the kept images.

4. **Repeat**: For multi-stage processing, repeat the workflow with the next stage
   ```bash
   python label.py --project "project_dir" --stage 2
   python train.py --project "project_dir" --stage 2 --epochs 50
   python cull.py --project "project_dir" --stage 2
   ```
   
5. **Automatic Stage Detection**: You can also omit the stage parameter to automatically use the latest stage
   ```bash
   python label.py --project "project_dir"
   python train.py --project "project_dir" --epochs 50
   python cull.py --project "project_dir"
   ```

## Technical Details

### Model Architecture
- Base model: Google's Vision Transformer (ViT) pretrained on ImageNet
- Modified with a classification head for binary classification (keep/cull)
- Input size: 224x224 RGB images
- Output: Binary classification (0=keep, 1=cull)

### Dataset
The `IMLCullDataset` class handles the image dataset:
- Reads image paths and labels from a CSV file
- Automatically resizes images to 224x224
- Applies standard ImageNet normalization
- Labels are mapped as: "keep": 0, "cull": 1

### Important Note

In this project, "culling" refers to the process of selecting images to **remove**, while "keeping" refers to images that should be **preserved**. The model predicts 0 for images to keep and 1 for images to cull. When the culling process runs, only images predicted as "keep" (prediction=0) are copied to the next stage directory.
