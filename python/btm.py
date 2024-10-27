import cv2
import torch
import numpy as np
from ultralytics import YOLO
import json
import time
from datetime import datetime
import os

# Load the YOLOv8 model
model = YOLO("models/best2.pt").to("cuda")

# Define the region of interest (ROI)
vertices = np.array([(563, 134), (646, 140), (591, 359), (257, 303)], dtype=np.int32)

# Set the actual road area in square meters
road_area_m2 = 1632.19

# Define vehicle sizes (length * width in meters)
car_size_m2 = 4.5 * 1.8  # Car
angkot_size_m2 = car_size_m2  # Angkot (same size as a car)
motorcycle_size_m2 = 2.0 * 0.8  # Motorcycle
truck_size_m2 = 7.5 * 2.5  # Truck (twice the size of a car)
bus_size_m2 = 12.0 * 2.5  # Bus (twice the size of a car)

# JSON file setup
json_filename = os.path.join('static/json', 'btm_data.json')

# Initialize last_save_time
last_save_time = time.time()

def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p1y
    return inside

def get_traffic_intensity(occupancy_percentage):
    if occupancy_percentage < 10:
        return "Lancar", (0, 255, 0)
    elif occupancy_percentage < 20:
        return "Sedang", (0, 255, 255)
    else:
        return "Padat", (0, 0, 255)

def save_to_json(occupancy_percentage):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {"Timestamp": current_time, "Occupancy Percentage": occupancy_percentage}

    # Read existing data
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as file:
            json_data = json.load(file)
    else:
        json_data = []

    # Append new data
    json_data.append(data)

    # Save back to JSON file
    with open(json_filename, 'w') as file:
        json.dump(json_data, file, indent=4)

def generate_frames():
    global last_save_time
    video_path = "static/videos/btm.mp4"  # or 0 for webcam
    cap = cv2.VideoCapture(video_path)

    # Initialize empty JSON file
    if not os.path.exists(json_filename):
        with open(json_filename, 'w') as file:
            json.dump([], file)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, imgsz=640)  # You can specify the size parameter

            # Initialize counters for each vehicle type
            car_count = 0
            angkot_count = 0
            motorcycle_count = 0
            truck_count = 0
            bus_count = 0
            total_occupied_area = 0

            # Process detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Check if the center of the box is inside the polygon
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    if point_inside_polygon(center_x, center_y, vertices):
                        # Get the class of the detected object
                        cls = int(box.cls[0])
                        class_name = model.names[cls]

                        if class_name == 'mobil':
                            car_count += 1
                            total_occupied_area += car_size_m2
                            color = (0, 255, 0)
                        elif class_name == 'angkot':
                            angkot_count += 1
                            total_occupied_area += angkot_size_m2
                            color = (0, 255, 255)  # Yellow for angkot
                        elif class_name == 'motor':
                            motorcycle_count += 1
                            total_occupied_area += motorcycle_size_m2
                            color = (255, 0, 0)  # Blue for motorcycles
                        elif class_name == 'truk':
                            truck_count += 1
                            total_occupied_area += truck_size_m2
                            color = (0, 0, 255)  # Red for trucks
                        elif class_name == 'bis':
                            bus_count += 1
                            total_occupied_area += bus_size_m2
                            color = (255, 165, 0)  # Orange for buses
                        else:
                            continue

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Add label
                        label = f"{class_name}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Calculate road occupancy
            occupancy_percentage = (total_occupied_area / road_area_m2) * 100

            # Save to JSON every 10 seconds
            current_time = time.time()
            if current_time - last_save_time >= 10:
                save_to_json(occupancy_percentage)
                last_save_time = current_time

            # Determine traffic intensity based on occupancy
            traffic_intensity, intensity_color = get_traffic_intensity(occupancy_percentage)

            # Draw the polygon on the frame
            cv2.polylines(frame, [vertices], True, (0, 255, 0), 2)

            # Initialize the vertical position for text display
            y_pos = 30
            text_gap = 40  # Gap between each line of text

            # Add text to the frame for all vehicle counts
            cv2.putText(frame, f"Mobil: {car_count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_pos += text_gap
            cv2.putText(frame, f"Angkot: {angkot_count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            y_pos += text_gap
            cv2.putText(frame, f"Motor: {motorcycle_count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            y_pos += text_gap
            cv2.putText(frame, f"Truk: {truck_count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            y_pos += text_gap
            cv2.putText(frame, f"Bis: {bus_count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
            y_pos += text_gap

            # Show the occupancy percentage and traffic intensity
            cv2.putText(frame, f"Kepadatan: {occupancy_percentage:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_pos += text_gap
            cv2.putText(frame, f"Traffic: {traffic_intensity}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, intensity_color, 2)


            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break

    cap.release()
