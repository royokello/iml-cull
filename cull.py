import os
import shutil
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

def find_album_directories(root_dir):
    """Find all album directories inside the root directory that have a src folder"""
    album_dirs = []
    
    # Check if the directory exists
    if not os.path.exists(root_dir):
        print(f"Root directory not found: {root_dir}")
        return album_dirs
    
    # List all subdirectories in the root directory
    for item in os.listdir(root_dir):
        album_path = os.path.join(root_dir, item)
        
        # Check if it's a directory
        if os.path.isdir(album_path):
            # Check if it has a src folder
            src_dir = os.path.join(album_path, 'src')
            
            if os.path.exists(src_dir):
                album_dirs.append(album_path)
    
    return album_dirs

def perform_culling(album_dir, model_path=None, device=None):
    """
    Uses a trained model to predict which images to cull.
    The function loads the model and then iterates through images in the 'src' folder.
    If the model predicts the label "keep" (i.e. prediction == 0),
    the image is copied to the 'src_culled' folder.
    
    Args:
        album_dir (str): Album directory containing the 'src' folder with images to process
        model_path (str): Path to the trained model file. If None, will look for it in the
                         parent directory of album_dir
        device (str): Device to run inference on ('cuda' or 'cpu'). Auto-detected if None.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    src_dir = os.path.join(album_dir, 'src')
    dest_dir = os.path.join(album_dir, 'src_culled')
    
    # If model_path is not provided, try to find it in the parent directory
    if model_path is None:
        # Try to find the model in the parent directory
        parent_dir = os.path.dirname(os.path.abspath(album_dir))
        model_path = os.path.join(parent_dir, 'cull_model.pth')
    
    if not os.path.exists(src_dir):
        print(f"Source directory not found: {src_dir}")
        return 0, 0  # Return counts of 0 for culled and kept
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}. Please ensure the model is trained.")
        return 0, 0  # Return counts of 0 for culled and kept
    
    print(f"Loading model from {model_path}")
    print(f"Using device: {device}")
    
    # Load model only once outside the function if processing multiple albums
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    total_images = len(os.listdir(src_dir))
    culled_count = 0
    kept_count = 0
    
    print(f"Processing {total_images} images in {src_dir}...")
    
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
        
        if prediction == 0:
            dest_path = os.path.join(dest_dir, img_name)
            shutil.copy(img_path, dest_path)
            print(f"Keeping {img_name} (copied to src_culled folder).")
            kept_count += 1
        else:
            print(f"Culling {img_name} (not copying to src_culled).")
            culled_count += 1
    
    print(f"\nProcessing complete for {os.path.basename(album_dir)}!")
    print(f"Total images processed: {total_images}")
    print(f"Images kept (copied to src_culled): {kept_count}")
    print(f"Images culled (not copied): {culled_count}")
    
    return culled_count, kept_count

def process_root_directory(root_dir, device=None):
    """
    Process all album directories in the root directory.
    
    Args:
        root_dir (str): Root directory containing multiple album folders
        device (str): Device to run inference on ('cuda' or 'cpu'). Auto-detected if None.
    """
    # Find all album directories
    album_dirs = find_album_directories(root_dir)
    
    if not album_dirs:
        print(f"No valid album directories found in {root_dir}")
        return
    
    print(f"Found {len(album_dirs)} album directories to process:")
    for album_dir in album_dirs:
        print(f"  - {album_dir}")
    
    # Model path in the root directory
    model_path = os.path.join(root_dir, 'cull_model.pth')
    
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}. Please ensure the model is trained.")
        return
    
    # Set up device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load model once for all albums
    print(f"Loading model from {model_path}...")
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Process each album
    total_culled = 0
    total_kept = 0
    
    for album_dir in album_dirs:
        album_name = os.path.basename(album_dir)
        print(f"\n{'='*50}")
        print(f"Processing album: {album_name}")
        print(f"{'='*50}")
        
        # Process this album's images
        src_dir = os.path.join(album_dir, 'src')
        dest_dir = os.path.join(album_dir, 'src_culled')
        
        if not os.path.exists(src_dir):
            print(f"Source directory not found: {src_dir}")
            continue
        
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        total_images = len(os.listdir(src_dir))
        culled_count = 0
        kept_count = 0
        
        print(f"Processing {total_images} images in {src_dir}...")
        
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
            
            if prediction == 0:
                dest_path = os.path.join(dest_dir, img_name)
                shutil.copy(img_path, dest_path)
                kept_count += 1
            else:
                culled_count += 1
        
        print(f"Album {album_name}: {kept_count} images kept, {culled_count} images culled")
        total_culled += culled_count
        total_kept += kept_count
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print("PROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total albums processed: {len(album_dirs)}")
    print(f"Total images processed: {total_culled + total_kept}")
    print(f"Total images kept (copied to src_culled): {total_kept}")
    print(f"Total images culled (not copied): {total_culled}")
    print(f"Keep rate: {total_kept/(total_culled + total_kept):.2%}")

if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Use a trained model to predict which images to cull')
    parser.add_argument('root_directory', type=str, 
                        help='Path to the root directory containing album folders to process')
    parser.add_argument('--album', type=str, default=None, 
                        help='Process only a specific album in the root directory')
    
    args = parser.parse_args()
    
    if args.album:
        # Process a specific album
        album_path = os.path.join(args.root_directory, args.album)
        model_path = os.path.join(args.root_directory, 'cull_model.pth')
        perform_culling(album_path, model_path)
    else:
        # Process all albums
        process_root_directory(args.root_directory)
