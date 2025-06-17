from ultralytics import YOLO
import cv2
import numpy as np
import time
import urllib.request
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Real-time Traffic Light Detection using IP Webcam')
    parser.add_argument('--ip', type=str, default='192.168.1.X', help='IP address of the Android IP Webcam app')
    parser.add_argument('--port', type=str, default='8080', help='Port of the Android IP Webcam app')
    parser.add_argument('--model', type=str, default='runs/detect/train2/weights/best.pt', help='Path to the trained YOLOv8 model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detection')
    parser.add_argument('--save', action='store_true', help='Save the output video')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video filename')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load the YOLOv8 model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    # IP Webcam settings
    url = f"http://{args.ip}:{args.port}/shot.jpg"
    print(f"Connecting to IP Webcam at {url}")
    
    # Initialize variables for FPS calculation
    prev_time = 0
    curr_time = 0
    fps_values = []
    
    # Initialize video writer if saving is enabled
    video_writer = None
    
    # Traffic light class information
    traffic_light_classes = {
        # Update these class indices based on your model's classes
        # Example: 0: 'red', 1: 'yellow', 2: 'green'
    }
    
    print("Starting real-time traffic light detection...")
    print("Press 'q' to quit, 's' to save a snapshot")
    
    try:
        # Test connection
        test_resp = urllib.request.urlopen(url)
        test_arr = np.array(bytearray(test_resp.read()), dtype=np.uint8)
        test_img = cv2.imdecode(test_arr, -1)
        
        if test_img is None:
            raise Exception("Failed to connect to IP Webcam")
            
        # Initialize video writer if saving is enabled
        if args.save:
            h, w, _ = test_img.shape
            video_writer = cv2.VideoWriter(
                args.output,
                cv2.VideoWriter_fourcc(*'mp4v'),
                20.0,  # FPS
                (w, h)
            )
            print(f"Saving output to {args.output}")
    
    except Exception as e:
        print(f"Connection error: {e}")
        print("Please make sure:")
        print("1. IP Webcam app is running on your Android device")
        print("2. Your computer and Android device are on the same network")
        print("3. The IP address and port are correct")
        print("4. The Android device has granted camera permissions to IP Webcam app")
        return
    
    while True:
        try:
            # Get image from IP Webcam
            img_resp = urllib.request.urlopen(url)
            img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            img = cv2.imdecode(img_arr, -1)
            
            if img is None:
                print("Failed to get image from IP Webcam")
                continue
            
            # Perform inference with confidence threshold
            results = model(img, conf=args.conf)
            
            # Process results
            annotated_frame = results[0].plot()
            
            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            fps_values.append(fps)
            
            # Calculate average FPS over the last 10 frames
            avg_fps = sum(fps_values[-10:]) / min(len(fps_values), 10)
            
            # Display FPS and detection info on the frame
            cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display detection information
            detections = results[0].boxes
            if len(detections) > 0:
                cv2.putText(annotated_frame, f"Detections: {len(detections)}", (20, 80), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display traffic light states if detected
                y_offset = 120
                for i, det in enumerate(detections):
                    cls_id = int(det.cls.item())
                    conf = float(det.conf.item())
                    
                    # Get class name if available
                    cls_name = traffic_light_classes.get(cls_id, f"Class {cls_id}")
                    
                    cv2.putText(annotated_frame, f"{cls_name}: {conf:.2f}", 
                              (20, y_offset + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the annotated frame
            cv2.imshow("Real-time Traffic Light Detection", annotated_frame)
            
            # Save frame to video if enabled
            if args.save and video_writer is not None:
                video_writer.write(annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save snapshot
                snapshot_name = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(snapshot_name, annotated_frame)
                print(f"Snapshot saved as {snapshot_name}")
                
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying in 3 seconds...")
            time.sleep(3)
    
    # Release resources
    if args.save and video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Detection stopped")

if __name__ == "__main__":
    main()