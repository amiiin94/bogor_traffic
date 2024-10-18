# Import libraries
from ultralytics import YOLO
import cv2
import math
from collections import defaultdict
import torch  # Import torch for GPU operations

# Check if CUDA is available and print the result
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"Current GPU: {torch.cuda.get_device_name()}")

# Path to the video file
video_path = 'pasaranyar.mp4'

# Start video capture from the file
cap = cv2.VideoCapture(video_path)

# Load the YOLO model
# Use .cuda() to move the model to GPU if available
model = YOLO("yolo-Weights/yolov5n.pt")
if torch.cuda.is_available():
    model = model.cuda()

# Define object classes for detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Initialize variables for car counting
car_count = 0
vehicle_ids = defaultdict(int)

# Read the first frame to get dimensions
success, img = cap.read()
if not success:
    print("Failed to read video")
    exit()

# Get frame dimensions
frame_height, frame_width = img.shape[:2]

# Define the counting line (horizontal line ending at the center of the frame)
line_y = frame_height // 2
line_x = frame_width // 2  # Center of the frame horizontally

# Line starts from the left side of the frame and ends at the center
line_start = (0, line_y)
line_end = (line_x, line_y)


# Infinite loop to continuously capture frames from the video
while True:
    # Read a frame from the video
    success, img = cap.read()

    # Break the loop if there are no more frames
    if not success:
        break

    # Convert the image to a CUDA tensor if GPU is available
    if torch.cuda.is_available():
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to('cuda').float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

    # Perform object detection using the YOLO model on the captured frame
    results = model(img, stream=True)

    # Convert img back to numpy array for OpenCV operations if it was on GPU
    if torch.cuda.is_available():
        img = results[0].orig_img

    # Draw the counting line
    cv2.line(img, line_start, line_end, (0, 255, 0), 2)

    # Iterate through the results of object detection
    for r in results:
        boxes = r.boxes

        # Iterate through each bounding box
        for box in boxes:
            # Extract coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw the bounding box on the frame
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # Calculate confidence score of the detection
            confidence = math.ceil((box.conf[0]*100))/100

            # Determine the class name of the detected object
            cls = int(box.cls[0])
            class_name = classNames[cls]

            # Check if the detected object is a vehicle (car, bus, or truck)
            if class_name in ['car', 'bus', 'truck', 'motorbike']:
                # Calculate the center of the bounding box
                center_y = (y1 + y2) // 2

                # Check if the vehicle has crossed the line
                if line_y - 5 <= center_y <= line_y + 5:
                    vehicle_id = f"{class_name}_{x1}_{y1}"
                    if vehicle_ids[vehicle_id] == 0:
                        car_count += 1
                        vehicle_ids[vehicle_id] = 1

            # Draw text indicating the class name on the frame
            org = [x1, y1 - 10]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 1
            cv2.putText(img, class_name, org, font, fontScale, color, thickness)

    # Draw the car count on the frame
    cv2.putText(img, f"Car Count: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with detected objects in a window named "Video"
    cv2.imshow('Video', img)

    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()