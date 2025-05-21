import torch
import torch.nn as nn
from transformers import ViTModel
from PIL import Image
from torchvision import transforms

class IMLCullModel(nn.Module):
    """
    ViT-based model for image culling.
    Uses a pre-trained Vision Transformer as the backbone and adds a classification head
    to predict whether an image should be kept (0) or culled (1).
    """
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224'):
        super(IMLCullModel, self).__init__()
        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        hidden_size = self.vit.config.hidden_size

        # Classification head for culling decision
        self.cull_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # Binary classification: 0 = keep, 1 = cull
        )

    def forward(self, pixel_values):
        """
        Forward pass through ViT and classification head.

        Args:
            pixel_values: Tensor of shape (batch, 3, 224, 224), already normalized.
        Returns:
            Tensor of logits for binary classification (keep/cull)
        """
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        logits = self.cull_head(cls_token)
        return logits
    
    def predict(self, image_path, device="cpu"):
        """
        Make prediction for a single image.
        
        Args:
            image_path: Path to the image file
            device: Device to use for inference
        Returns:
            Integer prediction: 0 = keep, 1 = cull
        """
        # Set up image transforms (same as in dataset.py)
        # Using ImageNet normalization statistics since the model was pre-trained on ImageNet
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        pixel_values = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        
        # Make prediction
        self.eval()
        self.to(device)
        with torch.no_grad():
            logits = self(pixel_values)
        
        prediction = torch.argmax(logits, dim=1).item()
        return prediction  # Returns 0 (keep) or 1 (cull)
    
    def predict_batch(self, batch_tensor, device="cpu"):
        """
        Make predictions for a batch of preprocessed images.
        
        Args:
            batch_tensor: Tensor of shape (batch_size, 3, 224, 224) containing preprocessed images
            device: Device to use for inference
        Returns:
            List of integer predictions: 0 = keep, 1 = cull
        """
        # Ensure model is in evaluation mode
        self.eval()
        self.to(device)
        
        # Move input to the correct device
        batch_tensor = batch_tensor.to(device)
        
        # Make predictions
        with torch.no_grad():
            logits = self(batch_tensor)
        
        # Get class predictions (0 = keep, 1 = cull)
        predictions = torch.argmax(logits, dim=1).tolist()
        return predictions
    
    @classmethod
    def from_pretrained(cls, model_path, device="cpu"):
        """
        Load a trained model from the given path
        
        Args:
            model_path: Path to the saved model weights
            device: Device to load the model on
        Returns:
            Loaded IMLCullModel instance
        """
        model = cls()
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise
        return model
