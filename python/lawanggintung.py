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

# Define ROI for jalan 1 as a polygon
roi_polygon_1 = np.array([ (485, 70), (418, 70), (362, 210), (666, 154)], np.int32)

# Define road dimensions and vehicle sizes
road_area_m2 = 85

vehicle_sizes = {
    'mobil': {'length': 4.5, 'width': 1.8, 'area': 4.5 * 1.8},
    'angkot': {'length': 4.5, 'width': 1.8, 'area': 4.5 * 1.8},
    'motor': {'length': 2.0, 'width': 0.8, 'area': 2.0 * 0.8},
    'truk': {'length': 7.5, 'width': 2.5, 'area': 7.5 * 2.5},
    'bis': {'length': 12.0, 'width': 2.5, 'area': 12.0 * 2.5}
}

colors = {
    'mobil': (0, 255, 0),
    'angkot': (0, 255, 255),
    'motor': (255, 0, 0),
    'truk': (0, 0, 255),
    'bis': (255, 165, 0)
}

# JSON file setup for jalan 1
json_filename_1 = os.path.join('static/json', 'lawanggintung_data.json')

# Initialize last_save_time
last_save_time = time.time()

def is_box_in_roi(box, roi_polygon):
    mask = np.zeros((800, 1280), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon], 1)
    
    box_mask = np.zeros((800, 1280), dtype=np.uint8)
    cv2.rectangle(box_mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 1, -1)
    
    intersection = cv2.bitwise_and(mask, box_mask)
    intersection_area = cv2.countNonZero(intersection)
    
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    overlap_ratio = intersection_area / box_area
    
    return overlap_ratio > 0.3

def get_traffic_level(occupancy_percent):
    OCCUPANCY_RINGAN = 15
    OCCUPANCY_SEDANG = 30

    if occupancy_percent <= OCCUPANCY_RINGAN:
        return "Ringan", (0, 255, 0)
    elif occupancy_percent <= OCCUPANCY_SEDANG:
        return "Sedang", (0, 255, 255)
    else:
        return "Padat", (0, 0, 255)

def save_to_json(occupancy_percentage, filename):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {"Timestamp": current_time, "Occupancy Percentage": occupancy_percentage}

    # Read existing data
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            json_data = json.load(file)
    else:
        json_data = []

    # Append new data
    json_data.append(data)

    # Save back to JSON file
    with open(filename, 'w') as file:
        json.dump(json_data, file, indent=4)

def generate_lawanggintung_frames():
    global last_save_time
    video_path = "static/videos/sukasari.mp4"  # or 0 for webcam
    cap = cv2.VideoCapture(video_path)

    # Initialize empty JSON file if it doesn't exist
    if not os.path.exists(json_filename_1):
        with open(json_filename_1, 'w') as file:
            json.dump([], file)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        visualization_frame = frame.copy()
        results = model(frame, imgsz=640, conf=0.3)

        vehicle_counts_1 = {vehicle_type: 0 for vehicle_type in vehicle_sizes.keys()}
        total_occupied_area_1 = 0

        # Draw ROI for jalan 1
        cv2.polylines(visualization_frame, [roi_polygon_1], True, (0, 255, 0), 2)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                color = colors.get(class_name, (128, 128, 128))
                cv2.rectangle(visualization_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(visualization_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if class_name in vehicle_sizes:
                    if is_box_in_roi([x1, y1, x2, y2], roi_polygon_1):
                        vehicle_counts_1[class_name] += 1
                        total_occupied_area_1 += vehicle_sizes[class_name]['area']

        occupancy_percentage_1 = (total_occupied_area_1 / road_area_m2) * 100
        traffic_level_1, level_color_1 = get_traffic_level(occupancy_percentage_1)

        # Display information for jalan 1
        y_pos = 30
        text_gap = 20

        y_pos += text_gap

        for vehicle_type, count in vehicle_counts_1.items():
            color = colors.get(vehicle_type, (255, 255, 255))
            cv2.putText(visualization_frame, f"{vehicle_type}: {count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += text_gap

        cv2.putText(visualization_frame, f"Occupancy: {occupancy_percentage_1:.1f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += text_gap
        cv2.putText(visualization_frame, f"Status: {traffic_level_1}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, level_color_1, 2)

        current_time = time.time()
        if current_time - last_save_time >= 10:
            save_to_json(occupancy_percentage_1, json_filename_1)  # Pass filename for ROI 1
            last_save_time = current_time

        ret, buffer = cv2.imencode('.jpg', visualization_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()