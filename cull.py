import os
import shutil
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

def perform_culling(root, device=None):
    """
    Uses a trained model from the 'cull_model' folder to predict which images to cull.
    The function loads the best model from 'cull_model/best_model.pth' and then iterates
    through images in the 'src' folder. If the model predicts the label "cull" (i.e. prediction == 1),
    the image is copied to the 'src_culled' folder.
    
    Args:
        root (str): Root directory containing 'src/', 'src_culled/', and 'cull_model/'.
        device (str): Device to run inference on ('cuda' or 'cpu'). Auto-detected if None.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    src_dir = os.path.join(root, 'src')
    dest_dir = os.path.join(root, 'src_culled')
    model_folder = os.path.join(root, 'cull_model')
    model_path = os.path.join(model_folder, 'best_model.pth')
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    if not os.path.exists(model_path):
        print("No best model found in the 'cull_model' folder. Please train a model first.")
        return
    
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    for img_name in os.listdir(src_dir):
        img_path = os.path.join(src_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            continue
        
        inputs = feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        
        if prediction == 1:
            dest_path = os.path.join(dest_dir, img_name)
            shutil.copy(img_path, dest_path)
            print(f"Copied {img_name} to culled folder based on prediction (cull).")
        else:
            print(f"Image {img_name} predicted as keep (prediction: {prediction}).")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python cull.py <root_directory>")
        exit(1)
    perform_culling(sys.argv[1])
