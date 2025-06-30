# YOLOv8 with Optimized CLAHE in YCbCr Space
import cv2
import yaml
import os
import glob
from ultralytics import YOLO
from pathlib import Path
import shutil 

def apply_clahe_ycrcb(image_path, clip_limit=4.0, tile_grid_size=(8, 8)):
    # Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the luminance (Y) channel of an image in the YCrCb color space.
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Warning: Could not read image at {image_path}")
        return None
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    y_clahe = clahe.apply(y)
    enhanced_ycrcb = cv2.merge([y_clahe, cr, cb])
    enhanced_bgr = cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2BGR)
    return enhanced_bgr

def preprocess_dataset(input_dir, output_dir, extensions=('*.jpg', '*.png', '*.jpeg')):
    # Applies the CLAHE preprocessing to all images in a directory.
    print(f"Starting preprocessing from '{input_dir}' to '{output_dir}'...")
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
            enhanced_image = apply_clahe_ycrcb(image_path)
            if enhanced_image is not None:
                output_image_path = output_subdir_images / Path(image_path).name
                cv2.imwrite(str(output_image_path), enhanced_image)
                label_path = Path(input_dir) / subdir / 'labels' / (Path(image_path).stem + '.txt')
                if label_path.exists():
                    output_label_path = output_subdir_labels / label_path.name
                    shutil.copy(str(label_path), str(output_label_path))
    
    print("Dataset preprocessing complete.")

def create_processed_yaml(original_yaml_path, new_yaml_path):
    # Creates a new data.yaml file for the processed dataset using relative paths.
    
    with open(original_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    data.pop('path', None)
    data['train'] = './train/images'
    data['val'] = './valid/images'
    data['test'] = './test/images'

    with open(new_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Created new YAML file at '{new_yaml_path}' with corrected relative paths.")

def main():
    original_dataset_dir = 'LISA-traffic-light-detection-3'
    processed_dataset_dir = 'LISA-processed-yolov8-3'
    
    if not os.path.exists(original_dataset_dir):
        print(f"Error: Original dataset not found at '{original_dataset_dir}'")
        return

    preprocess_dataset(original_dataset_dir, processed_dataset_dir)
    
    original_yaml = os.path.join(original_dataset_dir, 'data.yaml')
    new_yaml = os.path.join(processed_dataset_dir, 'data_clahe.yaml')
    create_processed_yaml(original_yaml, new_yaml)

    model = YOLO('yolov8n.pt') 

    print("\nStarting YOLOv8 training with parameters optimized for CPU...")
    try:
        results = model.train(
            data=new_yaml,
            epochs=10,
            imgsz=320,      
            batch=8,        
            workers=2,      
            patience=3,
            name='yolov8n_clahe_ycrcb_fast_cpu'
        )
        print("\nTraining complete.")
        print("Results saved to directory:", results.save_dir)

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

if __name__ == '__main__':
    main()