# YOLOv8 with Optimized CLAHE in YCbCr Space
import cv2
import yaml
import os
import glob
from ultralytics import YOLO
from pathlib import Path

def apply_clahe_ycrcb(image_path, clip_limit=4.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the
    luminance (Y) channel of an image in the YCrCb color space.
    This method is inspired by the paper "CLAHE-Based Low-Light Image
    Enhancement for Robust Object Detection in Overhead Power Transmission System"
    (Yuan et al., 2023), which found an optimized clip limit of 4.0 to be effective.

    Args:
        image_path (str): The path to the input image.
        clip_limit (float): The contrast limiting threshold.
        tile_grid_size (tuple): The size of the grid for histogram equalization.

    Returns:
        numpy.ndarray: The enhanced image in BGR format.
    """
    # Read the image in BGR format
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Warning: Could not read image at {image_path}")
        return None

    # Convert the image from BGR to YCrCb color space
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    # Split the YCrCb image into its Y, Cr, and Cb channels
    y, cr, cb = cv2.split(img_ycrcb)

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE to the L (lightness) channel
    y_clahe = clahe.apply(y)

    # Merge the CLAHE-enhanced Y channel with the original Cr and Cb channels
    enhanced_ycrcb = cv2.merge([y_clahe, cr, cb])

    # Convert the enhanced image back to BGR color space
    enhanced_bgr = cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2BGR)

    return enhanced_bgr

def preprocess_dataset(input_dir, output_dir, extensions=('*.jpg', '*.png', '*.jpeg')):
    """
    Applies the CLAHE preprocessing to all images in a directory and saves them
    to an output directory, maintaining the original structure.

    Args:
        input_dir (str): Path to the root directory of the original dataset.
        output_dir (str): Path to the directory where preprocessed images will be saved.
    """
    print(f"Starting preprocessing from '{input_dir}' to '{output_dir}'...")
    for subdir in ['train', 'valid', 'test']:
        # Create corresponding subdirectories in the output folder
        output_subdir_images = Path(output_dir) / subdir / 'images'
        output_subdir_labels = Path(output_dir) / subdir / 'labels'
        output_subdir_images.mkdir(parents=True, exist_ok=True)
        output_subdir_labels.mkdir(parents=True, exist_ok=True)

        input_subdir_images = Path(input_dir) / subdir / 'images'

        # Process images
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(str(input_subdir_images / ext)))
        
        print(f"Found {len(image_files)} images in '{subdir}'...")

        for image_path in image_files:
            # Apply enhancement
            enhanced_image = apply_clahe_ycrcb(image_path)

            if enhanced_image is not None:
                # Save the enhanced image
                output_image_path = output_subdir_images / Path(image_path).name
                cv2.imwrite(str(output_image_path), enhanced_image)

                # Copy the corresponding label file
                label_path = Path(input_dir) / subdir / 'labels' / (Path(image_path).stem + '.txt')
                if label_path.exists():
                    output_label_path = output_subdir_labels / label_path.name
                    os.system(f'copy "{label_path}" "{output_label_path}"') # Use 'cp' on Linux/macOS
    
    print("Dataset preprocessing complete.")

def create_processed_yaml(original_yaml_path, new_yaml_path, processed_data_dir):
    """
    Creates a new data.yaml file for the processed dataset.
    
    Args:
        original_yaml_path (str): Path to the original data.yaml file.
        new_yaml_path (str): Path to save the new yaml file.
        processed_data_dir (str): Path to the root of the preprocessed dataset.
    """
    with open(original_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update paths to point to the new preprocessed dataset location
    data['path'] = os.path.abspath(processed_data_dir)
    data['train'] = os.path.join(processed_data_dir, 'train', 'images')
    data['val'] = os.path.join(processed_data_dir, 'valid', 'images')
    data['test'] = os.path.join(processed_data_dir, 'test', 'images')

    with open(new_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Created new YAML file at '{new_yaml_path}'")

def main():
    # --- 1. Preprocessing Step ---
    # Define paths
    original_dataset_dir = 'LISA-traffic-light-detection'
    processed_dataset_dir = 'LISA-processed-yolov8'
    
    # Check if the original dataset exists
    if not os.path.exists(original_dataset_dir):
        print(f"Error: Original dataset not found at '{original_dataset_dir}'")
        print("Please ensure the LISA dataset is in the correct directory.")
        return

    # Preprocess the entire dataset
    preprocess_dataset(original_dataset_dir, processed_dataset_dir)
    
    # Create the new YAML configuration file for the processed dataset
    original_yaml = os.path.join(original_dataset_dir, 'data.yaml')
    new_yaml = os.path.join(processed_dataset_dir, 'data_clahe.yaml')
    create_processed_yaml(original_yaml, new_yaml, processed_dataset_dir)


    # --- 2. YOLOv8 Training Step ---
    # Load the YOLOv8n model
    model = YOLO('yolov8n.pt') 

    # Train the model using the preprocessed dataset
    print("\nStarting YOLOv8 training with the preprocessed dataset...")
    try:
        results = model.train(
            data=new_yaml,
            epochs=20,
            imgsz=640,
            batch=16,
            patience=3, # Early stopping patience
            name='yolov8n_clahe_ycrcb'
        )
        print("\nTraining complete.")
        print("Results saved to directory:", results.save_dir)

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        print("Please ensure the 'ultralytics' library is installed and paths are correct.")

if __name__ == '__main__':
    main()