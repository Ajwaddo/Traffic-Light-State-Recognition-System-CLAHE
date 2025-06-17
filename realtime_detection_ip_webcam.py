from ultralytics import YOLO
import cv2
import numpy as np
import time
import requests
import urllib.request

def main():
    # Load the YOLOv8 model
    model = YOLO('runs/detect/train2/weights/best.pt')  # Load your trained model
    
    # IP Webcam settings
    # Replace with your Android IP Webcam app's IP address and port
    ip_address = "192.168.100.6"  # Replace X with your device's IP
    port = "8080"
    url = f"http://{ip_address}:{port}/shot.jpg"
    
    # Initialize variables for FPS calculation
    prev_time = 0
    curr_time = 0
    
    print("Starting real-time traffic light detection...")
    print(f"Connecting to IP Webcam at {url}")
    print("Press 'q' to quit")
    
    while True:
        try:
            # Get image from IP Webcam
            img_resp = urllib.request.urlopen(url)
            img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            img = cv2.imdecode(img_arr, -1)
            
            if img is None:
                print("Failed to get image from IP Webcam")
                continue
            
            # Perform inference
            results = model(img)
            
            # Process results
            annotated_frame = results[0].plot()
            
            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # Display FPS on the frame
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the annotated frame
            cv2.imshow("Real-time Traffic Light Detection", annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying in 3 seconds...")
            time.sleep(3)
    
    # Release resources
    cv2.destroyAllWindows()
    print("Detection stopped")

if __name__ == "__main__":
    main()