import os
import csv
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch import nn, optim
from typing import Tuple, List, Dict, Optional, Union, Any, Callable
from utils import find_latest_stage
from model import IMLCullModel
from dataset import IMLCullDataset

def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset_size: int,
    device: str
) -> float:
    model.train()
    running_loss = 0.0
    
    for batch in dataloader:
        pixel_values, labels = batch[0], batch[1]
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(pixel_values=pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * pixel_values.size(0)
        
    return running_loss / dataset_size

def validate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    dataset_size: int,
    device: str
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values, labels = batch[0], batch[1]
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            logits = model(pixel_values=pixel_values)
            loss = criterion(logits, labels)
            running_loss += loss.item() * pixel_values.size(0)
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total
    return running_loss / dataset_size, accuracy

def train_model(
    project_dir: str,
    num_epochs: int,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 16,
    validation_split: float = 0.25,
    device: str = "cpu",
    stage: Optional[int] = None,
    resume: bool = False
) -> Optional[torch.nn.Module]:
    try:
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
    model_initialized_from_scratch = True
    if resume:
        if stage > 1: # Check if there's a previous stage to load from
            previous_stage_number = stage - 1
            previous_model_path = os.path.join(project_dir, f"stage_{previous_stage_number}_cull_model.pth")
            if os.path.exists(previous_model_path):
                print(f"Attempting to load model weights from previous stage: {previous_model_path}")
                try:
                    model = IMLCullModel() # Initialize model structure
                    model.load_state_dict(torch.load(previous_model_path, map_location=device))
                    model.to(device)
                    print(f"Successfully loaded model weights from {previous_model_path}.")
                    model_initialized_from_scratch = False
                except Exception as e:
                    print(f"Error loading weights from {previous_model_path}: {e}. Initializing with default pre-trained weights.")
            else:
                print(f"Previous stage model {previous_model_path} not found. Initializing with default pre-trained weights.")
        else:
            print(f"Current stage is {stage}. No previous stage model to load. Initializing with default pre-trained weights.")
    
    if model_initialized_from_scratch:
        print("Initializing model with default pre-trained weights (google/vit-base-patch16-384).")
        model = IMLCullModel(pretrained_model_name='google/vit-base-patch16-384')
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

        if val_accuracy > best_val_accuracy:
            if abs(val_loss - train_loss) > (val_loss/2):
                epochs_no_improve += 1
                continue

            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_file)
            print("Saving ...")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, type=str)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--epochs', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=4)
    
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
        stage=args.stage,
    )
