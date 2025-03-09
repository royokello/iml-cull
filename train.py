import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torch import nn, optim
from transformers import ViTForImageClassification, ViTFeatureExtractor
import logging

# Set up logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class CullDataset(Dataset):
    def __init__(self, root, feature_extractor):
        self.root = root
        self.image_dir = os.path.join(root, 'src')
        self.feature_extractor = feature_extractor
        self.labels = {}
        csv_path = os.path.join(root, 'cull_labels.csv')
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
        return pixel_values, label

def train_model(root, num_epochs=20, batch_size=8, learning_rate=1e-4, patience=3, validation_split=0.2, device="cpu"):
    # Initialize feature extractor and model.
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2)
    model.to(device)
    
    # Create dataset and split into training and validation.
    dataset = CullDataset(root, feature_extractor)
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    # Create model saving folder.
    model_folder = os.path.join(root, "cull_model")
    if not os.path.exists(model_folder):
         os.makedirs(model_folder)
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for pixel_values, labels in train_loader:
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
            for pixel_values, labels in val_loader:
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                loss = criterion(logits, labels)
                running_val_loss += loss.item() * pixel_values.size(0)
        val_loss = running_val_loss / val_size
        
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Check for improvement in validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model_save_path = os.path.join(model_folder, "best_model.pth")
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Validation loss improved. Model saved to {model_save_path}.")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    logging.info("Training complete.")
    return model

def predict(root, model, image_path, device="cpu"):
    from transformers import ViTFeatureExtractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction  # Returns 0 (keep) or 1 (cull)

if __name__ == '__main__':
    import sys
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(sys.argv) < 2:
        print("Usage: python train.py <root_directory>")
        exit(1)
    train_model(sys.argv[1], device=device)
