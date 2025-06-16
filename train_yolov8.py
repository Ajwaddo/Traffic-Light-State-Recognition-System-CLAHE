from ultralytics import YOLO

# Define the model size (n, s, m, l, x)
model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8 nano model

# Train the model using your dataset
results = model.train(
    data='/Users/muhdajwd/Library/Mobile Documents/com~apple~CloudDocs/MMU/Degree/DELTA/TRIM 2/FYP2/Traffic-Light-State-Recognition-System-CLAHE/LISA-traffic-light-detection/data.yaml',
    epochs=20,           # Number of training epochs
    imgsz=640,            # Image size
    batch=16,             # Batch size
    patience=20,          # Early stopping patience
    save=True,            # Save training results
    device='mps'            # GPU device (use 'cpu' if no GPU available)
)

# Validate the model on the validation set
results = model.val()

# Export the model to ONNX format (optional)
model.export(format='onnx')