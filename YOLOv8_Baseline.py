import os
from ultralytics import YOLO

def main():
    """
    This script trains a baseline YOLOv8n model on the original, unprocessed
    LISA traffic light dataset. The purpose of this script is to establish a
    performance benchmark against which the models trained on CLAHE-enhanced
    data can be compared.
    """

    # --- 1. Dataset Configuration ---
    # Define the path to the original dataset's configuration file.
    # This assumes the script is run from the same root directory as the
    # 'LISA-traffic-light-detection' folder.
    original_dataset_yaml = os.path.join('LISA-traffic-light-detection', 'data.yaml')

    # Verify that the dataset configuration file exists before starting.
    if not os.path.exists(original_dataset_yaml):
        print(f"Error: Original dataset YAML file not found at '{original_dataset_yaml}'")
        print("Please ensure the 'LISA-traffic-light-detection' folder and its 'data.yaml' file are present.")
        return

    # --- 2. YOLOv8 Training Step ---
    # Load the YOLOv8n model with pre-trained weights.
    model = YOLO('yolov8n.pt') 

    # Train the model using the original (unprocessed) dataset.
    print(f"\nStarting YOLOv8 baseline training with the original dataset: {original_dataset_yaml}")
    
    try:
        results = model.train(
            data=original_dataset_yaml,
            epochs=10,
            imgsz=320,
            batch=16,
            patience=3, # Early stopping patience
            name='yolov8n_baseline_no_clahe' # Descriptive name for the output folder
        )
        print("\nBaseline training complete.")
        print("Results saved to directory:", results.save_dir)

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        print("Please ensure the 'ultralytics' library is installed and that dataset paths are correct.")

if __name__ == '__main__':
    main()