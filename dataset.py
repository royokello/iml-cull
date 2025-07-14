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
            transforms.Resize((384, 384)),
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
            # The 'label' column should now directly contain 0 (Keep) or 1 (Cull).
            # The 'tag' column is ignored for training purposes by this Dataset class.
            for row in reader:
                try:
                    # Handle the image path from CSV - this can be absolute, relative, or just a filename
                    image_path_from_csv = row['image_path']
                    
                    # We'll use this as our key in self.labels
                    image_key = image_path_from_csv # Store the original value from CSV
                    label_str = row['label']
                    label_value = int(label_str)
                    if label_value not in [0, 1]:
                        print(f"Warning: Invalid label value '{label_value}' for image '{image_path_from_csv}' in {label_file_path}. Expected 0 or 1. Skipping.")
                        continue
                    self.labels[image_key] = label_value
                except ValueError:
                    print(f"Warning: Non-integer label value '{label_str}' for image '{image_path_from_csv}' in {label_file_path}. Skipping.")
                    continue
                except KeyError as e:
                    print(f"Warning: CSV row missing expected key ({e}) in {label_file_path}. Row: {row}. Skipping.")
                    continue
        self.images = list(self.labels.keys()) # This will now be a list of image_paths
    
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
        # Get the image path from our stored list
        img_path_from_csv = self.images[idx]
        
        # Handle the path correctly - if it's a relative path or just a filename
        # join it with image_dir, otherwise use it as is if it's an absolute path
        if os.path.isabs(img_path_from_csv) and os.path.exists(img_path_from_csv):
            # If it's an absolute path and it exists, use it directly
            img_path = img_path_from_csv
        else:
            # Otherwise, treat it as relative to image_dir
            # First check if it's just a basename or a relative path
            img_basename = os.path.basename(img_path_from_csv)
            img_path = os.path.join(self.image_dir, img_basename)
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms to the image
        pixel_values = self.transform(image)
        
        label = self.labels[self.images[idx]] # Use the exact same key from our images list
        return pixel_values, label, img_path  # Return image path for identification
