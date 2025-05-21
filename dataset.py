import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class IMLCullDataset(Dataset):
    """
    Dataset class for IML Cull project.
    Loads images and labels from image directory and label file.
    """
    def __init__(self, image_dir_path, label_file_path):
        """
        Initialize the dataset.
        
        Args:
            image_dir_path: Path to the directory containing images
            label_file_path: Path to the CSV file containing labels
        """
        self.labels = {}
        self.image_dir = image_dir_path
        
        # Set up image transforms (ViT model expects 224x224 images)
        # Using ImageNet normalization statistics since the model was pre-trained on ImageNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Check if paths exist
        if not os.path.exists(image_dir_path):
            raise ValueError(f"Image directory not found: {image_dir_path}")
            
        if not os.path.exists(label_file_path):
            raise ValueError(f"Label file not found: {label_file_path}")
            
        with open(label_file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            # Map "cull" to 1, "keep" to 0.
            for row in reader:
                self.labels[row['image_name']] = 1 if row['label'].lower() == 'cull' else 0
        self.images = list(self.labels.keys())
    
    def __len__(self):
        """Return the number of images in the dataset"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            tuple: (pixel_values, label, img_path)
        """
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms to the image
        pixel_values = self.transform(image)
        
        label = self.labels[img_name]
        return pixel_values, label, img_path  # Return image path for identification
