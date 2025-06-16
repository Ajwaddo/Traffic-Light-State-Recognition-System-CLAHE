from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('runs/detect/train2/weights/best.pt')  # Load a pretrained YOLOv8 nano model

# Perform inference on an image
results = model('LISA-traffic-light-detection/valid/images/dayClip5-00095_jpg.rf.348295c77939e270c1d731e6d0199920.jpg')

#Process results
for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoint = result.keypoints
    probs = result.probs

    print(f"Detected {len(boxes)} objects")

    
    result.save("result.jpg")

result.show()
