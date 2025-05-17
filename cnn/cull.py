import os
import shutil
import torch
from PIL import Image
from torchvision import transforms
from dataset import resize_and_pad_square
from model import create_model
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def setup_transforms():
    """Create standard transforms for image preprocessing"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict_image(model, image_path, transform, device):
    """Predict whether to keep or cull a single image"""
    try:
        image = Image.open(image_path).convert('RGB')
        image, _ = resize_and_pad_square(image, target_size=224)
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            prediction = torch.argmax(outputs, dim=1).item()
        
        return prediction == 1  # 1 means "keep", 0 means "cull"
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        return False

def perform_culling(input_dir, source_dir='src', device=None):
    """
    Process images in the source directory and copy "keep" images to a cull directory.
    
    Args:
        input_dir: Root directory containing the source folder and model
        source_dir: Name of the source directory (default: 'src')
        device: Device to run inference on ('cuda' or 'cpu')
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup paths
    src_dir = os.path.join(input_dir, source_dir)
    cull_dir = os.path.join(input_dir, 'cull')
    model_path = os.path.join(input_dir, 'cull_model.pth')

    # Reset crop directory (delete if exists, then recreate)
    if os.path.exists(cull_dir):
        shutil.rmtree(cull_dir)
    os.makedirs(cull_dir, exist_ok=True)
    
    # Validate directories
    if not os.path.exists(src_dir):
        logging.error(f"Source directory not found: {src_dir}")
        return
    
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return
    
    # Load model
    logging.info(f"Loading model from {model_path}")
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Setup transforms
    transform = setup_transforms()
    
    # Process images
    total_images = len(os.listdir(src_dir))
    kept_count = 0
    culled_count = 0
    
    logging.info(f"Processing {total_images} images in {src_dir}...")
    
    for img_name in os.listdir(src_dir):
        img_path = os.path.join(src_dir, img_name)
        
        if predict_image(model, img_path, transform, device):
            # Copy image to cull directory if predicted as "keep"
            dest_path = os.path.join(cull_dir, img_name)
            shutil.copy(img_path, dest_path)
            logging.info(f"Keeping {img_name}")
            kept_count += 1
        else:
            logging.info(f"Culling {img_name}")
            culled_count += 1
    
    # Print summary
    logging.info("\nProcessing complete!")
    logging.info(f"Total images processed: {total_images}")
    logging.info(f"Images kept (copied to cull folder): {kept_count}")
    logging.info(f"Images culled (not copied): {culled_count}")
    if total_images > 0:
        logging.info(f"Keep rate: {kept_count/total_images:.2%}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Root directory containing the source folder with images')
    p.add_argument('--source', default='src', help='Name of the source directory containing images (default: src)')
    args = p.parse_args()
    
    perform_culling(args.input, args.source)

if __name__ == '__main__':
    main()