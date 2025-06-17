# Real-time Traffic Light Detection with Android IP Webcam

This guide explains how to use your trained YOLOv8 model for real-time traffic light detection using an Android phone as an IP webcam.

## Prerequisites

1. A trained YOLOv8 model for traffic light detection
2. An Android smartphone
3. Computer with Python installed
4. Both devices connected to the same Wi-Fi network

## Setup Instructions

### 1. Install IP Webcam App on Android

1. Download and install the "IP Webcam" app from Google Play Store
   - [IP Webcam on Google Play](https://play.google.com/store/apps/details?id=com.pas.webcam)

2. Open the app and configure settings:
   - Set video preferences (resolution, quality)
   - Enable "Start server on app launch" if desired

### 2. Start the IP Webcam Server

1. In the IP Webcam app, scroll down and tap "Start server"
2. The app will display an IP address like `http://192.168.1.X:8080`
3. **Note this IP address and port number** - you'll need it for the detection script

### 3. Run the Detection Script

#### Basic Script

Use `realtime_detection_ip_webcam.py` for a simple implementation:

1. Edit the script to update the IP address:
   ```python
   ip_address = "192.168.1.X"  # Replace X with your device's IP
   ```

2. Run the script:
   ```bash
   python realtime_detection_ip_webcam.py
   ```

#### Advanced Script

Use `advanced_realtime_detection.py` for more features:

```bash
python advanced_realtime_detection.py --ip 192.168.1.X --port 8080 --conf 0.25 --save --output traffic_detection.mp4
```

Command-line arguments:
- `--ip`: IP address of your Android device
- `--port`: Port number (usually 8080)
- `--model`: Path to your trained model (default: runs/detect/train2/weights/best.pt)
- `--conf`: Confidence threshold (default: 0.25)
- `--save`: Enable to save the output video
- `--output`: Output video filename

## Troubleshooting

### Connection Issues

1. **Cannot connect to IP Webcam**:
   - Ensure both devices are on the same Wi-Fi network
   - Check if the IP address and port are correct
   - Try accessing the webcam stream in a browser using the same URL

2. **Slow performance**:
   - Lower the resolution in the IP Webcam app
   - Reduce the frame rate
   - Use a more powerful computer for processing

3. **Detection not working**:
   - Verify your model path is correct
   - Try increasing the confidence threshold
   - Ensure the camera has good lighting conditions

### Common Errors

1. **URLError**: Check your network connection and IP address
2. **Model not found**: Verify the path to your trained model
3. **OpenCV errors**: Make sure OpenCV is properly installed

## Tips for Better Performance

1. Position your phone with a clear view of traffic lights
2. Use a phone mount or tripod for stability
3. Ensure good lighting conditions
4. Close other apps on both devices to free up resources
5. If using in a vehicle, connect your phone to a power source

## Keyboard Controls

- Press 'q' to quit the detection
- Press 's' to save a snapshot (advanced script only)