import os
import shutil
import torch
from PIL import Image
from torchvision import transforms
from model import IMLCullModel
from utils import find_latest_stage

# No need for a separate function, we'll use find_latest_stage directly

def perform_culling(project_dir, stage=None, batch_size=64):
    """
    Uses a trained model to predict which images to cull.
    The function loads the model and then processes images in batches from the stage directory.
    If the model predicts the label "keep" (i.e. prediction == 0),
    the image is copied to the next stage directory.
    
    Args:
        project_dir (str): Project directory containing stage folders
        stage (int): Stage number to process (if None, will use the latest stage)
        batch_size (int): Number of images to process at once (default: 64)
        
    Returns:
        Tuple of (culled_count, kept_count)
    """
    # Set device automatically
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get the stage number if not provided
    try:
        if stage is None:
            stage = find_latest_stage(project_dir)
            print(f"Using latest stage: {stage}")
            
        # Set up the stage directory path
        stage_dir_path = os.path.join(project_dir, f"stage_{stage}")
        
        # Check if the directory exists
        if not os.path.exists(stage_dir_path):
            raise ValueError(f"Stage directory not found: {stage_dir_path}")
    except ValueError as e:
        print(f"Error: {e}")
        return 0, 0  # Return counts of 0 for culled and kept
    
    # Set up the paths
    src_dir = stage_dir_path
    next_stage = stage + 1
    dest_dir = os.path.join(project_dir, f"stage_{next_stage}")
    model_path = os.path.join(project_dir, f"stage_{stage}_cull_model.pth")
    
    # Reset the destination directory if it exists
    if os.path.exists(dest_dir):
        print(f"Removing existing stage_{next_stage} directory...")
        shutil.rmtree(dest_dir)
    
    # Create the destination directory
    os.makedirs(dest_dir)
    
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}. Please ensure the model is trained.")
        return 0, 0  # Return counts of 0 for culled and kept
    
    print(f"Loading model from {model_path}")
    print(f"Using device: {device}")
    
    # Load model using IMLCullModel
    try:
        model = IMLCullModel.from_pretrained(model_path, device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0, 0
    
    total_images = len(os.listdir(src_dir))
    culled_count = 0
    kept_count = 0
    
    print(f"Processing {total_images} images in {src_dir}...")
    
    # Get all valid image files
    image_files = [f for f in os.listdir(src_dir) if not os.path.isdir(os.path.join(src_dir, f))]
    total_images = len(image_files)
    
    print(f"Processing {total_images} images in batches of {batch_size}...")
    
    # Create the image transform (same as in the model.py and dataset.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process images in batches
    batch_count = (total_images + batch_size - 1) // batch_size  # Ceiling division
    
    for batch_idx in range(batch_count):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        current_batch_size = end_idx - start_idx
        
        print(f"Processing batch {batch_idx+1}/{batch_count} ({current_batch_size} images)...")
        
        # Prepare the batch
        batch_tensors = []
        batch_names = []
        
        # Load and preprocess each image in the batch
        for idx in range(start_idx, end_idx):
            img_name = image_files[idx]
            img_path = os.path.join(src_dir, img_name)
            
            try:
                image = Image.open(img_path).convert("RGB")
                tensor = transform(image)
                batch_tensors.append(tensor)
                batch_names.append(img_name)
            except Exception as e:
                print(f"Error processing image {img_name}: {e}")
                # Skip this image
                continue
        
        if not batch_tensors:  # Skip if all images in batch failed to load
            continue
        
        # Stack tensors into a batch and get predictions
        batch_tensor = torch.stack(batch_tensors)
        predictions = model.predict_batch(batch_tensor, device)
        
        # Process predictions and copy files accordingly
        for img_name, prediction in zip(batch_names, predictions):
            img_path = os.path.join(src_dir, img_name)
            
            if prediction == 0:  # 0 = keep
                dest_path = os.path.join(dest_dir, img_name)
                shutil.copy(img_path, dest_path)
                kept_count += 1
                print(f"Keeping {img_name} (copied to stage_{next_stage} folder).")
            else:  # 1 = cull
                culled_count += 1
                print(f"Culling {img_name} (not copying to stage_{next_stage}).")
    
    processed_count = kept_count + culled_count
    print(f"Processed {processed_count}/{total_images} images.")
    if processed_count < total_images:
        print(f"Warning: {total_images - processed_count} images could not be processed due to errors.")

    
    print(f"Culling complete. Kept {kept_count} images, culled {culled_count} images.")
    print(f"Kept images are saved in stage_{next_stage} directory.")
    return culled_count, kept_count

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Use a trained model to predict which images to cull')
    parser.add_argument('--project', required=True, type=str, 
                        help='Path to the project directory containing stage folders')
    parser.add_argument('--stage', type=int, default=None, 
                        help='Stage number to process (if not provided, will use the latest stage)')
    parser.add_argument('--batch-size', type=int, default=512, 
                        help='Number of images to process at once (default: 512)')
    
    args = parser.parse_args()
    
    # Call the perform_culling function with the provided arguments
    perform_culling(args.project, args.stage, args.batch_size)
