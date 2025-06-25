from ultralytics import YOLO


model = YOLO('yolov8n.pt')  


results = model.train(
    data='LISA-traffic-light-detection/data.yaml',
    epochs=20,           # Number of training epochs
    imgsz=640,            # Image size
    batch=16,             # Batch size
    patience=20,          # Early stopping patience
    save=True,            # Save training results
    device='cpu'            # GPU device 
)

# Validate the model on the validation set
results = model.val()


model.export(format='onnx')