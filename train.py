import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from PIL import Image
from torch import nn, optim
from transformers import ViTForImageClassification, ViTImageProcessor
import logging

# Set up logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class CullDataset(Dataset):
    def __init__(self, album_path, feature_extractor):
        self.album_path = album_path
        self.image_dir = os.path.join(album_path, 'src')
        self.feature_extractor = feature_extractor
        self.labels = {}
        csv_path = os.path.join(album_path, 'cull_labels.csv')
        
        if not os.path.exists(csv_path):
            logging.warning(f"No labels file found at {csv_path}")
            self.images = []
            return
            
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            # Map "cull" to 1, "keep" to 0.
            for row in reader:
                self.labels[row['image_name']] = 1 if row['label'].lower() == 'cull' else 0
        self.images = list(self.labels.keys())
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # Remove batch dimension.
        label = self.labels[img_name]
        return pixel_values, label, img_path  # Return image path for identification


def find_album_directories(root_dir):
    """Find all album directories inside the root directory that have labeled data"""
    album_dirs = []
    
    # Check if the directory exists
    if not os.path.exists(root_dir):
        logging.error(f"Root directory not found: {root_dir}")
        return album_dirs
    
    # List all subdirectories in the root directory
    for item in os.listdir(root_dir):
        album_path = os.path.join(root_dir, item)
        
        # Check if it's a directory
        if os.path.isdir(album_path):
            # Check if it has the expected structure (src folder and cull_labels.csv)
            src_dir = os.path.join(album_path, 'src')
            labels_file = os.path.join(album_path, 'cull_labels.csv')
            
            if os.path.exists(src_dir) and os.path.exists(labels_file):
                album_dirs.append(album_path)
    
    return album_dirs


def train_model(root_dir, num_epochs, batch_size=64, learning_rate=1e-3, patience=16, validation_split=0.25, device="cpu"):
    """
    Train a model using all labeled albums found in the root directory.
    
    Args:
        root_dir: Root directory containing multiple album folders
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        patience: Early stopping patience
        validation_split: Fraction of data to use for validation
        device: Device to use for training (cpu or cuda)
    """
    # Find all album directories
    album_dirs = find_album_directories(root_dir)
    
    if not album_dirs:
        logging.error(f"No valid album directories found in {root_dir}")
        return None
    
    logging.info(f"Found {len(album_dirs)} album directories with labeled data:")
    for album_dir in album_dirs:
        logging.info(f"  - {album_dir}")
    
    # Set up model file paths in the root directory
    model_file = os.path.join(root_dir, "cull_model.pth")
    val_loss_file = os.path.join(root_dir, "cull_val_loss.txt")
    
    # Check if we should resume training from a previous run
    start_epoch = 0
    best_val_loss = float("inf")
    
    if os.path.exists(model_file):
        # Model exists, try to resume
        try:
            # Load the best validation loss if available
            if os.path.exists(val_loss_file):
                with open(val_loss_file, "r") as f:
                    best_val_loss = float(f.read().strip())
                    logging.info(f"Loaded best validation loss: {best_val_loss:.8f}")
            
            # Check if there's a saved epoch number
            if os.path.exists(os.path.join(root_dir, "cull_epoch.txt")):
                with open(os.path.join(root_dir, "cull_epoch.txt"), "r") as f:
                    start_epoch = int(f.read().strip())
                    logging.info(f"Found existing model at epoch {start_epoch}. Resuming training from epoch {start_epoch + 1}.")
        except (ValueError, FileNotFoundError) as e:
            logging.warning(f"Error loading previous training state: {e}")
            start_epoch = 0
            best_val_loss = float("inf")
    
    # Initialize feature extractor and model.
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2, ignore_mismatched_sizes=True)
    
    # Load model weights if found
    if os.path.exists(model_file):
        try:
            model.load_state_dict(torch.load(model_file, map_location=device))
            logging.info(f"Loaded model weights from {model_file}")
        except Exception as e:
            logging.warning(f"Error loading model weights: {e}")
    
    model.to(device)
    
    # Create datasets for each album directory and concatenate them
    datasets = []
    for album_dir in album_dirs:
        try:
            dataset = CullDataset(album_dir, feature_extractor)
            if len(dataset) > 0:
                datasets.append(dataset)
                logging.info(f"Loaded {len(dataset)} labeled images from {album_dir}")
            else:
                logging.warning(f"No labeled images found in {album_dir}")
        except Exception as e:
            logging.warning(f"Error loading dataset from {album_dir}: {e}")
    
    if not datasets:
        logging.error("No valid datasets found. Exiting.")
        return None
    
    # Combine all datasets into one
    if len(datasets) == 1:
        combined_dataset = datasets[0]
    else:
        combined_dataset = ConcatDataset(datasets)
    
    # Log dataset statistics
    total_size = len(combined_dataset)
    logging.info(f"Combined dataset size: {total_size} images")
    
    # Split into training and validation
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])
    
    logging.info(f"Training set: {train_size} images, Validation set: {val_size} images")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    epochs_no_improve = 0
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_train_loss = 0.0
        for batch in train_loader:
            pixel_values, labels = batch[0], batch[1]  # Unpack the batch (ignore image paths)
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * pixel_values.size(0)
        train_loss = running_train_loss / train_size
        
        # Validation phase.
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values, labels = batch[0], batch[1]  # Unpack the batch (ignore image paths)
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                loss = criterion(logits, labels)
                running_val_loss += loss.item() * pixel_values.size(0)
        val_loss = running_val_loss / val_size
        
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.8f} - Val Loss: {val_loss:.8f}")
        
        # Save current epoch number
        with open(os.path.join(root_dir, "cull_epoch.txt"), "w") as f:
            f.write(str(epoch + 1))
        
        # Check for improvement in validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save model
            torch.save(model.state_dict(), model_file)
            
            # Save the best validation loss
            with open(val_loss_file, "w") as f:
                f.write(str(best_val_loss))
                
            logging.info(f"Validation loss improved. Model saved to {model_file}.")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    logging.info("Training complete.")
    return model

def predict(model, image_path, device="cpu"):
    from transformers import ViTImageProcessor
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction  # Returns 0 (keep) or 1 (cull)

def load_model(model_path, device="cpu"):
    """Load a trained model from the given path"""
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

if __name__ == '__main__':
    import sys
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Train a ViT model for image culling across all labeled albums')
    parser.add_argument('root_directory', type=str, help='Path to the root directory containing multiple album folders')
    parser.add_argument('--epochs', type=int, default=256, help='Number of epochs to train for (default: 256)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=16, help='Early stopping patience (default: 16)')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    train_model(
        args.root_directory,
        num_epochs=args.epochs, 
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device=device
    )
