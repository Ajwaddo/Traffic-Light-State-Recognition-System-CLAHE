# YOLOv7 with CLAHE and Dehazing
import cv2
import numpy as np
import yaml
import os
import glob
from pathlib import Path
import shutil # Import the shutil library for robust file copying

# Note: YOLOv7 is typically trained via command line with its official repository.
# This script focuses on the preprocessing and provides the training command.

def get_dark_channel(image, window_size=15):
    """
    Calculates the dark channel of an image, a key component in dehazing.
    This is part of the methodology inspired by "Edge-Computing-Facilitated Nighttime
    Vehicle Detection Investigations With CLAHE-Enhanced Images" (Lashkov, Yuan, & Zhang, 2023).

    Args:
        image (numpy.ndarray): Input BGR image.
        window_size (int): Size of the patch for dark channel calculation.

    Returns:
        numpy.ndarray: The dark channel of the image.
    """
    b, g, r = cv2.split(image)
    min_channel = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def get_atmospheric_light(image, dark_channel, percentile=0.001):
    """
    Estimates the atmospheric light from the brightest pixels in the dark channel.
    """
    image_size = image.shape[0] * image.shape[1]
    num_pixels = int(max(np.floor(image_size * percentile), 1))
    
    dark_vec = dark_channel.reshape(image_size)
    image_vec = image.reshape(image_size, 3)
    
    indices = dark_vec.argsort()
    indices = indices[image_size - num_pixels:]
    
    atm_sum = np.zeros([1, 3])
    for idx in indices:
        atm_sum += image_vec[idx]
        
    atmospheric_light = atm_sum / num_pixels
    return atmospheric_light

def estimate_transmission(image, atmospheric_light, omega=0.95, window_size=15):
    """
    Estimates the transmission map of the image.
    """
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
    
    q = mean_a * I + mean_b
    return q

def recover_dehazed_image(image, transmission, atmospheric_light, t0=0.1):
    """
    Recovers the dehazed image using the haze imaging model.
    """
    transmission = cv2.max(transmission, t0)
    
    J = np.empty(image.shape, image.dtype)
    for ind in range(0, 3):
        J[:, :, ind] = (image[:, :, ind] - atmospheric_light[0, ind]) / transmission + atmospheric_light[0, ind]
    
    return np.clip(J, 0, 255).astype(np.uint8)

def apply_dehazing_and_auto_clahe(image_path):
    """
    Applies a combined dehazing and auto-clipped CLAHE enhancement.
    This method simulates the advanced preprocessing from Lashkov et al. (2023).

    Args:
        image_path (str): The path to the input image.

    Returns:
        numpy.ndarray: The enhanced image in BGR format.
    """
    # 1. Dehazing to reduce glare/halo effects
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Warning: Could not read image at {image_path}")
        return None
        
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_float = img_bgr.astype('float64') / 255
    
    dark_channel = get_dark_channel(img_float)
    atmospheric_light = get_atmospheric_light(img_float, dark_channel)
    raw_transmission = estimate_transmission(img_float, atmospheric_light)
    
    # Refine transmission map with guided filter
    refined_transmission = guided_filter(gray_img / 255, raw_transmission, r=60, eps=0.0001)
    
    dehazed_image = recover_dehazed_image(img_bgr, refined_transmission, atmospheric_light * 255)

    # 2. Auto-Clipped CLAHE on the luminance channel
    img_lab = cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    
    mean_brightness = np.mean(l)
    clip_limit = max(1.0, 8.0 - (mean_brightness / 32.0))

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    enhanced_lab = cv2.merge([l_clahe, a, b])
    final_enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return final_enhanced_image

def preprocess_dataset_yolov7(input_dir, output_dir, extensions=('*.jpg', '*.png', '*.jpeg')):
    """Applies the dehazing and CLAHE preprocessing to a dataset."""
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
                    # --- FIX 2: Using shutil.copy for robustness ---
                    shutil.copy(str(label_path), str(output_label_path))
    
    print("YOLOv7 dataset preprocessing complete.")

def create_processed_yaml_v7(original_yaml_path, new_yaml_path, processed_data_dir):
    """Creates a new YAML for the processed dataset for YOLOv7."""
    with open(original_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # --- FIX 1: Use relative paths for the dataset YAML file ---
    # The YOLOv7 repo expects paths to be relative to the repository root,
    # so providing a full absolute path in the command line is the most robust way.
    # This function now sets up the structure correctly, and the command below
    # will use the absolute path to this new YAML file.
    data['train'] = str(Path(processed_data_dir).resolve() / 'train' / 'images')
    data['val'] = str(Path(processed_data_dir).resolve() / 'valid' / 'images')
    data['test'] = str(Path(processed_data_dir).resolve() / 'test' / 'images')
    
    # Remove the 'path' key to avoid confusion
    data.pop('path', None)

    with open(new_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Created new YAML file for YOLOv7 at '{new_yaml_path}' with corrected absolute paths.")

def main():
    # --- 1. Preprocessing Step ---
    original_dataset_dir = 'LISA-traffic-light-detection'
    processed_dataset_dir_v7 = 'LISA-processed-yolov7'

    if not os.path.exists(original_dataset_dir):
        print(f"Error: Original dataset not found at '{original_dataset_dir}'")
        return

    preprocess_dataset_yolov7(original_dataset_dir, processed_dataset_dir_v7)
    
    original_yaml = os.path.join(original_dataset_dir, 'data.yaml')
    new_yaml_v7 = os.path.join(processed_dataset_dir_v7, 'data_clahe_dehaze.yaml')
    create_processed_yaml_v7(original_yaml, new_yaml_v7, processed_dataset_dir_v7)

    # --- 2. YOLOv7 Training Step ---
    print("\nTo train YOLOv7, run the following command from within the yolov7 repository:")
    
    # Construct the training command using yolov7-tiny as the nano equivalent.
    training_command = (
        f"python train.py --workers 8 --device 0 --batch-size 16 "
        f"--data \"{os.path.abspath(new_yaml_v7)}\" --img 640 640 --cfg cfg/training/yolov7-tiny.yaml "
        f"--weights yolov7-tiny.pt --name yolov7_clahe_dehaze --hyp data/hyp.scratch.tiny.yaml "
        f"--epochs 20"
        # Note: Early stopping in YOLOv7 is handled via its internal mechanisms based on validation metrics,
        # rather than a direct 'patience' argument.
    )
    
    print("\n--- YOLOv7 Training Command ---")
    print(training_command)
    print("---------------------------------\n")
    print("Note: Ensure you are in the yolov7 repository directory and have downloaded 'yolov7-tiny.pt' weights.")

if __name__ == '__main__':
    main()