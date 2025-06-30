# YOLOv5 with Standard CLAHE
import cv2
import yaml
import os
import glob
from ultralytics import YOLO
from pathlib import Path
import shutil # Import the shutil library for robust file copying

def apply_standard_clahe(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies a standard Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to a grayscale version of the image. This simulates the preprocessing step
    of a two-stage detector as described in "Traffic Light Recognition Using
    Deep Learning in Adverse Conditions".

    Args:
        image_path (str): The path to the input image.
        clip_limit (float): The contrast limiting threshold.
        tile_grid_size (tuple): Size of the grid for histogram equalization.

    Returns:
        numpy.ndarray: The enhanced image in BGR format.
    """
    # Read the image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Warning: Could not read image at {image_path}")
        return None

    # Convert image to grayscale
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE to the grayscale image
    enhanced_gray = clahe.apply(gray_img)
    
    # Convert the enhanced grayscale image back to a 3-channel BGR image
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    return enhanced_bgr

def preprocess_dataset_yolov5(input_dir, output_dir, extensions=('*.jpg', '*.png', '*.jpeg')):
    """
    Applies the standard CLAHE preprocessing to a dataset for YOLOv5 training.
    """
    print(f"Starting YOLOv5 preprocessing from '{input_dir}' to '{output_dir}'...")
    for subdir in ['train', 'valid', 'test']:
        output_subdir_images = Path(output_dir) / subdir / 'images'
        output_subdir_labels = Path(output_dir) / subdir / 'labels'
        output_subdir_images.mkdir(parents=True, exist_ok=True)
        output_subdir_labels.mkdir(parents=True, exist_ok=True)

        input_subdir_images = Path(input_dir) / subdir / 'images'

        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(str(input_subdir_images / ext)))
            
        print(f"Found {len(image_files)} images in '{subdir}'...")

        for image_path in image_files:
            enhanced_image = apply_standard_clahe(image_path)
            if enhanced_image is not None:
                output_image_path = output_subdir_images / Path(image_path).name
                cv2.imwrite(str(output_image_path), enhanced_image)

                label_path = Path(input_dir) / subdir / 'labels' / (Path(image_path).stem + '.txt')
                if label_path.exists():
                    output_label_path = output_subdir_labels / label_path.name
                    # --- FIX 2: Using shutil.copy for robustness ---
                    shutil.copy(str(label_path), str(output_label_path))

    print("YOLOv5 dataset preprocessing complete.")

def create_processed_yaml_v5(original_yaml_path, new_yaml_path, processed_data_dir):
    """Creates a new YAML for the YOLOv5 processed dataset."""
    with open(original_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # --- FIX 1: Use relative paths for the dataset YAML file ---
    data.pop('path', None) # Remove the absolute path key
    data['train'] = './train/images'
    data['val'] = './valid/images'
    data['test'] = './test/images'
    # --- END FIX 1 ---

    with open(new_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Created new YAML file for YOLOv5 at '{new_yaml_path}' with corrected relative paths.")


def main():
    # --- 1. Preprocessing Step ---
    original_dataset_dir = 'LISA-traffic-light-detection'
    processed_dataset_dir_v5 = 'LISA-processed-yolov5'
    
    if not os.path.exists(original_dataset_dir):
        print(f"Error: Original dataset not found at '{original_dataset_dir}'")
        return

    preprocess_dataset_yolov5(original_dataset_dir, processed_dataset_dir_v5)
    
    original_yaml = os.path.join(original_dataset_dir, 'data.yaml')
    new_yaml_v5 = os.path.join(processed_dataset_dir_v5, 'data_clahe_standard.yaml')
    create_processed_yaml_v5(original_yaml, new_yaml_v5, processed_dataset_dir_v5)

    # --- 2. YOLOv5 Training Step ---
    # Load the YOLOv5n model
    # Note: Using the ultralytics library which supports YOLOv5 as well.
    model = YOLO('yolov5n.pt') 

    # Train the model
    print("\nStarting YOLOv5 training with the preprocessed dataset...")
    try:
        results = model.train(
            data=new_yaml_v5,
            epochs=20,
            imgsz=640,
            batch=16,
            patience=3, # Early stopping patience
            name='yolov5n_standard_clahe'
        )
        print("\nTraining complete.")
        print("Results saved to directory:", results.save_dir)

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

if __name__ == '__main__':
    main()