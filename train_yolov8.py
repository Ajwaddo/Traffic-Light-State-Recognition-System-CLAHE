from ultralytics import YOLO

# Define the model size (n, s, m, l, x)
model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8 nano model

# Train the model using your dataset
results = model.train(
    data='LISA-traffic-light-detection/data.yaml',
    epochs=10,           # Number of training epochs
    imgsz=640,            # Image size
    batch=16,             # Batch size
    patience=3,          # Early stopping patience
    save=True,            # Save training results
    device='cpu'            # GPU device (use 'cpu' if no GPU available)
)

# Validate the model on the validation set
results = model.val()

# Export the model to ONNX format (optional)
model.export(format='onnx')