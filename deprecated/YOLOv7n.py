import cv2
import numpy as np
import yaml
import os
import glob
from pathlib import Path
import shutil

# Note: YOLOv7 is typically trained via command line with its official repository.
# This script focuses on the preprocessing and provides the training command.

def get_dark_channel(image, window_size=15):
    """Calculates the dark channel of an image."""
    b, g, r = cv2.split(image)
    min_channel = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def get_atmospheric_light(image, dark_channel, percentile=0.001):
    """Estimates the atmospheric light."""
    image_size = image.shape[0] * image.shape[1]
    num_pixels = int(max(np.floor(image_size * percentile), 1))
    dark_vec = dark_channel.reshape(image_size)
    image_vec = image.reshape(image_size, 3)
    indices = dark_vec.argsort()[-num_pixels:]
    atm_sum = np.sum(image_vec[indices], axis=0)
    return atm_sum / num_pixels

def estimate_transmission(image, atmospheric_light, omega=0.95, window_size=15):
    """Estimates the transmission map."""
    norm_image = image / atmospheric_light
    transmission = 1 - omega * get_dark_channel(norm_image, window_size)
    return transmission

def guided_filter(I, p, r, eps):
    """Fast guided filter implementation."""
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    return mean_a * I + mean_b

def recover_dehazed_image(image, transmission, atmospheric_light, t0=0.1):
    """Recovers the dehazed image."""
    transmission = cv2.max(transmission, t0)
    J = np.empty(image.shape, image.dtype)
    for ind in range(3):
        J[:, :, ind] = (image[:, :, ind] - atmospheric_light[ind]) / transmission + atmospheric_light[ind]
    return np.clip(J, 0, 255).astype(np.uint8)

def apply_dehazing_and_auto_clahe(image_path):
    """Applies combined dehazing and auto-clipped CLAHE."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return None
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_float = img_bgr.astype('float64') / 255
    dark_channel = get_dark_channel(img_float)
    atmospheric_light = get_atmospheric_light(img_float, dark_channel)
    raw_transmission = estimate_transmission(img_float, atmospheric_light)
    refined_transmission = guided_filter(gray_img / 255, raw_transmission, r=60, eps=0.0001)
    dehazed_image = recover_dehazed_image(img_bgr, refined_transmission, atmospheric_light * 255)
    img_lab = cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    mean_brightness = np.mean(l)
    clip_limit = max(1.0, 8.0 - (mean_brightness / 32.0))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    enhanced_lab = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def preprocess_dataset_yolov7(input_dir, output_dir, extensions=('*.jpg', '*.png', '*.jpeg')):
    """Preprocesses a dataset for YOLOv7 training."""
    print(f"Starting YOLOv7 preprocessing from '{input_dir}' to '{output_dir}'...")
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
            enhanced_image = apply_dehazing_and_auto_clahe(image_path)
            if enhanced_image is not None:
                output_image_path = output_subdir_images / Path(image_path).name
                cv2.imwrite(str(output_image_path), enhanced_image)
                label_path = Path(input_dir) / subdir / 'labels' / (Path(image_path).stem + '.txt')
                if label_path.exists():
                    output_label_path = output_subdir_labels / label_path.name
                    shutil.copy(str(label_path), str(output_label_path))
    print("YOLOv7 dataset preprocessing complete.")

def create_processed_yaml_v7(original_yaml_path, new_yaml_path, processed_data_dir):
    """Creates a new YAML for the processed dataset for YOLOv7."""
    # --- FIX ---
    # Load the original YAML to preserve any existing keys if needed,
    # but explicitly define the necessary keys for training.
    try:
        with open(original_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        data = {}

    # Define the required keys for the YOLOv7 training script
    data['train'] = str(Path(processed_data_dir).resolve() / 'train' / 'images')
    data['val'] = str(Path(processed_data_dir).resolve() / 'valid' / 'images')
    data['test'] = str(Path(processed_data_dir).resolve() / 'test' / 'images')
    
    # Explicitly set the number of classes and class names
    data['nc'] = 3
    data['names'] = ['go', 'stop', 'warning']
    
    # Remove the 'path' key to avoid confusion
    data.pop('path', None)

    with open(new_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Created new YAML file for YOLOv7 at '{new_yaml_path}'")
    # --- END FIX ---


def main():
    original_dataset_dir = 'LISA-traffic-light-detection-3'
    processed_dataset_dir_v7 = 'LISA-processed-yolov7-3' # Using the directory from your error message

    if not os.path.exists(original_dataset_dir):
        print(f"Error: Original dataset not found at '{original_dataset_dir}'")
        return

    preprocess_dataset_yolov7(original_dataset_dir, processed_dataset_dir_v7)
    
    original_yaml = os.path.join(original_dataset_dir, 'data.yaml')
    new_yaml_v7 = os.path.join(processed_dataset_dir_v7, 'data_clahe_dehaze.yaml')
    create_processed_yaml_v7(original_yaml, new_yaml_v7, processed_dataset_dir_v7)

    print("\nTo train YOLOv7, run the following command from within the yolov7 repository:")
    
    # Your training command from the previous turn, now with the correct YAML
    training_command = (
        f"python train.py --workers 2 --device cpu --batch-size 8 "
        f"--data \"{os.path.abspath(new_yaml_v7)}\" --img 320 320 --cfg cfg/training/yolov7-tiny.yaml "
        f"--weights yolov7-tiny.pt --name yolov7_clahe_dehaze_fast_cpu --hyp data/hyp.scratch.tiny.yaml "
        f"--epochs 10" # As per your command
    )
    
    print("\n--- YOLOv7 Training Command (Optimized for CPU) ---")
    print(training_command)
    print("--------------------------------------------------\n")
    print("Ensure you are in the yolov7 repository directory and have downloaded 'yolov7-tiny.pt' weights.")


if __name__ == '__main__':
    main()