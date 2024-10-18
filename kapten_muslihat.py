import cv2
import math
from collections import defaultdict
import torch
from ultralytics import YOLO

def generate_frames():
    # Load the YOLO model
    model = YOLO("yolo-Weights/best.pt")
    if torch.cuda.is_available():
        model = model.cuda()

    # Start video capture
    cap = cv2.VideoCapture('pasaranyar.mp4')

    # Initialize variables for car counting
    car_count = 0
    vehicle_ids = defaultdict(int)

    # Read the first frame to get dimensions
    success, img = cap.read()
    if not success:
        print("Failed to read video")
        return

    # Get frame dimensions
    frame_height, frame_width = img.shape[:2]

    # Define the counting line
    line_y = frame_height // 2
    line_x = frame_width // 2
    line_start = (0, line_y)
    line_end = (line_x, line_y)

    while True:
        success, img = cap.read()
        if not success:
            break

        # Perform object detection
        results = model(img, stream=True)

        # Draw the counting line
        cv2.line(img, line_start, line_end, (0, 255, 0), 2)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract coordinates and draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

                # Get class name and confidence
                cls = int(box.cls[0])
                class_name = model.names[cls]
                confidence = math.ceil((box.conf[0]*100))/100

                # Count vehicles
                if class_name in ['car', 'bus', 'truck', 'motorbike']:
                    center_y = (y1 + y2) // 2
                    if line_y - 5 <= center_y <= line_y + 5:
                        vehicle_id = f"{class_name}_{x1}_{y1}"
                        if vehicle_ids[vehicle_id] == 0:
                            car_count += 1
                            vehicle_ids[vehicle_id] = 1

                # Draw class name
                cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        # Draw the car count
        cv2.putText(img, f"Car Count: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()