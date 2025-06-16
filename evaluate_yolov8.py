from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('runs/detect/train2/weights/best.pt')  # Load a pretrained YOLOv8 nano model

# Perform inference on an image
results = model.val(
    data='LISA-traffic-light-detection/data.yaml',
    split='test'
)

print(results)
