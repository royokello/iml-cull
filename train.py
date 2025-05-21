import os
import csv
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn, optim
from typing import Tuple, List, Dict, Optional, Union, Any, Callable
from utils import find_latest_stage
from model import IMLCullModel
from dataset import IMLCullDataset

def train_epoch(model: torch.nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             criterion: torch.nn.Module, 
             optimizer: torch.optim.Optimizer, 
             dataset_size: int, 
             device: str) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        dataloader: DataLoader for the training dataset
        criterion: Loss function
        optimizer: Optimizer for updating model weights
        dataset_size: Size of the training dataset
        device: Device to use for training (cpu or cuda)
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    for batch in dataloader:
        pixel_values, labels = batch[0], batch[1]  # Unpack the batch (ignore image paths)
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(pixel_values=pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * pixel_values.size(0)
        
    return running_loss / dataset_size

def validate_epoch(model: torch.nn.Module, 
                 dataloader: torch.utils.data.DataLoader, 
                 criterion: torch.nn.Module, 
                 dataset_size: int, 
                 device: str) -> Tuple[float, float]:
    """
    Validate the model for one epoch.
    
    Args:
        model: The neural network model
        dataloader: DataLoader for the validation dataset
        criterion: Loss function
        dataset_size: Size of the validation dataset
        device: Device to use for validation (cpu or cuda)
        
    Returns:
        Tuple of (average validation loss, accuracy) for the epoch
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values, labels = batch[0], batch[1]  # Unpack the batch (ignore image paths)
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            logits = model(pixel_values=pixel_values)
            loss = criterion(logits, labels)
            running_loss += loss.item() * pixel_values.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total
    return running_loss / dataset_size, accuracy

def train_model(project_dir: str, 
             num_epochs: int, 
             batch_size: int = 64, 
             learning_rate: float = 1e-3, 
             patience: int = 16, 
             validation_split: float = 0.25, 
             device: str = "cpu", 
             stage: Optional[int] = None) -> Optional[torch.nn.Module]:
    """
    Train a model using labeled data from the project directory.
    
    Args:
        project_dir: Path to the project directory containing stage folders
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        patience: Early stopping patience
        validation_split: Fraction of data to use for validation
        device: Device to use for training (cpu or cuda)
        stage: Specific stage to use for training (if None, will use the latest stage)
    """
    # Detect the stage before creating the dataset
    try:
        # If stage is not provided, try to find the latest stage
        if stage is None:
            try:
                stage = find_latest_stage(project_dir)
                print(f"Using latest stage: {stage}")
            except ValueError as e:
                print(f"Error: No stage specified and {str(e)}")
                return None
        
        # Set up paths
        stage_dir = f"stage_{stage}"
        image_dir_path = os.path.join(project_dir, stage_dir)
        label_file_path = os.path.join(project_dir, f"stage_{stage}_cull_labels.csv")
        
        # Check if paths exist
        if not os.path.exists(image_dir_path):
            raise ValueError(f"Stage directory not found: {image_dir_path}")
        
        if not os.path.exists(label_file_path):
            raise ValueError(f"Label file not found: {label_file_path}")
        
        # Create dataset with explicit paths
        dataset = IMLCullDataset(image_dir_path, label_file_path)
        if len(dataset) == 0:
            raise ValueError(f"No labeled images found in {image_dir_path}")
        
        print(f"Using stage_{stage} from project {project_dir} with {len(dataset)} labeled images")
    except Exception as e:
        print(f"Error setting up dataset: {e}")
        return None
    
    # Set up model file paths in the project directory
    model_file = os.path.join(project_dir, f"stage_{stage}_cull_model.pth")
    epoch_log_file = os.path.join(project_dir, f"stage_{stage}_epoch_log.csv")
    
    # Delete previous model files if they exist
    for file_path in [model_file]:
        if os.path.exists(file_path):
            print(f"Removing existing model file: {file_path}")
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete file {file_path}: {e}")
                
    # Create or overwrite the epoch log file with headers
    with open(epoch_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy'])
        print(f"Created new epoch log file: {epoch_log_file}")
    
    # Always start from scratch
    start_epoch = 0
    best_val_accuracy = 0.0
    
    # Initialize model
    model = IMLCullModel(pretrained_model_name='google/vit-base-patch16-224')
    model.to(device)
    
    # Log dataset statistics
    total_size = len(dataset)
    print(f"Dataset size: {total_size} images")
    
    # Split into training and validation
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training set: {train_size} images, Validation set: {val_size} images")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    epochs_no_improve = 0
    
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, train_size, device)
        val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, val_size, device)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.8f} - Val Loss: {val_loss:.8f} - Val Accuracy: {val_accuracy:.4f}")
        
        # Append to the epoch log file
        with open(epoch_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, val_accuracy])
        
        # Check for improvement in validation accuracy.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            # Save model
            torch.save(model.state_dict(), model_file)
            print(f"Validation accuracy improved to {best_val_accuracy:.4f}. Model saved to {model_file}.")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    print("Training complete.")
    return model

if __name__ == '__main__':
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Train a ViT model for image culling across all labeled projects')
    parser.add_argument('--project', required=True, type=str, help='Path to the project directory containing stage folders')
    parser.add_argument('--stage', type=int, help='Stage number to use for training (if not provided, will use the latest stage)')
    parser.add_argument('--epochs', type=int, default=256, help='Number of epochs to train for (default: 256)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience (default: 16)')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    train_model(
        args.project,
        num_epochs=args.epochs, 
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device=device,
        stage=args.stage
    )
