# iml-cull
A simple image culling toolkit with manual labeling built on Flask, AI-assisted culling recommendations based on Google ViT, and automated image management.

## Overview

iml-cull is a toolkit for image culling, which helps photographers and image editors efficiently manage large collections of photos by identifying and separating lower-quality images. The toolkit consists of three primary modules:

1. **Label Module** - Manual image labeling interface
2. **Train Module** - Training a custom Vision Transformer (ViT) model
3. **Cull Module** - Automated image culling based on the trained model

## Project Structure

To use iml-cull, create a directory structure as follows:

```
root_directory/
├── src/            # Source images to process
├── src_culled/     # Directory for culled images (created automatically)
├── cull_model/     # Directory for trained model (created automatically)
└── cull_labels.csv # Labels file (created during labeling)
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
python label.py <root_directory>
```

Where `<root_directory>` is the path to your project folder containing the `src` directory with your images.

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

The train module uses your labeled data to train a Vision Transformer (ViT) model to automatically classify images.

**Usage:**
```
python train.py <root_directory>
```

Where `<root_directory>` is the same root directory used in the labeling step.

**Features:**
- Uses Google's ViT (Vision Transformer) architecture
- Automatically splits data into training and validation sets
- Implements early stopping to prevent overfitting
- Saves the best model based on validation performance
- Supports both CPU and GPU training (automatically detects CUDA)

**Parameters:**
- Default training for 20 epochs
- Batch size of 8
- Learning rate of 1e-4
- Patience of 3 epochs for early stopping
- 20% validation split

The trained model is saved to `<root_directory>/cull_model/best_model.pth`.

### 3. Cull Module (`cull.py`)

The cull module uses your trained model to automatically identify and separate images that should be culled.

**Usage:**
```
python cull.py <root_directory>
```

Where `<root_directory>` is the same root directory used in previous steps.

**Features:**
- Loads the best model from training
- Processes all images in the `src` directory
- Copies images predicted as "cull" to the `src_culled` directory
- Leaves original files untouched
- Provides console output for each prediction

**Workflow:**
1. Train a model using `train.py`
2. Run `cull.py` to automatically process images
3. Review the culled images in the `src_culled` directory
4. Manually verify the model's decisions

## Complete Workflow Example

```

# Step 1: Install dependencies
pip install flask torch transformers Pillow

# Step 2: Create your project structure with images in src/ folder
mkdir -p my_project/src
cp my_images/* my_project/src/

# Step 3: Manual labeling
python label.py my_project

# Step 4: Train the model
python train.py my_project

# Step 5: Automatic culling
python cull.py my_project

```

After these steps, you'll find the images identified for culling in `my_project/src_culled/`.

## Notes

- The training process requires sufficient labeled data for accurate results
- GPU acceleration is recommended for faster training but not required
- The quality of culling predictions depends on the consistency of your manual labeling
