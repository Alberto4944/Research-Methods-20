import cv2 as cv
import os
from ultralytics import YOLO

# 1. Load your best model and video
model = YOLO("runs/detect/train-5/weights/best.pt")
video_path = "videos/test.mov"  # Change to your video name
output_dir = "dataset_export"

# Create output directories for images and labels
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/labels", exist_ok=True)

cap = cv.VideoCapture(video_path)
frame_count = 0
saved_count = 0

print("Processing video and generating automatic labels...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # To keep your dataset manageable, sample every 2nd or 3rd frame 
    # (Fast rallies change drastically frame-by-frame)
    if frame_count % 2 == 0:
        frame_name = f"frame_{saved_count:05d}"
        image_path = f"{output_dir}/images/{frame_name}.jpg"
        label_path = f"{output_dir}/labels/{frame_name}.txt"
        
        # Run YOLO prediction at a decent confidence threshold
        results = model.predict(frame, conf=0.25, verbose=False)
        
        # Save the raw image frame
        cv.imwrite(image_path, frame)
        
        # Write the tracking box if YOLO found the ball
        with open(label_path, "w") as f:
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    # Get normalized xywh coordinates (class, x_center, y_center, width, height)
                    # Class '0' is your table tennis ball
                    cls = int(box.cls[0])
                    xywh = box.xywhn[0].cpu().numpy() 
                    f.write(f"{cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n")
        
        saved_count += 1
    frame_count += 1

cap.release()
print(f"Done! Extracted {saved_count} frames and labels into the '{output_dir}' folder.")