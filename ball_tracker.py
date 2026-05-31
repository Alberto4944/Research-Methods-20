import cv2 as cv
import os
from ultralytics import YOLO

file_type_choice = 0;
files = []
num = 1

ball_model = YOLO("best_models/best.pt")

def track_ball(frame):
    h, w = frame.shape[:2]
    if w > 960:
        frame = cv.resize(frame, (960, int(h * 960 / w)))
    if w < 960:
        frame = cv.resize(frame, (960, int(h/w*960)))
    results = ball_model.predict(frame, conf=0.3, verbose=False)
    ball_x, ball_y = -1.0, -1.0
    if len(results[0].boxes) > 0:
        box = results[0].boxes.xywh[0]
        ball_x, ball_y = float(box[0]), float(box[1])
        cv.circle(frame, (int(ball_x), int(ball_y)), 50, (0,255,0), -1)
    return frame;    

while file_type_choice != 1 and file_type_choice != 2 and file_type_choice != 3:
    file_type_choice = int(input("Image (1), Video (2), or Live Camera (3): "))

if file_type_choice == 1:
    for file in os.listdir("images"):
        if file.endswith((".jpg", ".jpeg", ".png")):
            files.append(file)
            print(f"{num}. {file}")
            num+=1
    file_choice = int(input("Select an image by typing the file number: "))
    image_path = f"images/{files[file_choice-1]}"
    cv.imshow("Ball Tracking", track_ball(cv.imread(image_path)))
    cv.waitKey(0)

elif file_type_choice == 2:
    for file in os.listdir("videos"):
        if file.endswith((".mp4", ".mov", ".MOV")):
            files.append(file)
            print(f"{num}. {file}")
            num+=1
    file_choice = int(input("Select a video by typing the file number: "))
    video_path = f"videos/{files[file_choice-1]}"
    cap = cv.VideoCapture(video_path)
        
else:
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

if file_type_choice == 2 or file_type_choice == 3:
    while cap.isOpened():
        ret, frame = cap.read();
        if not ret:
            print("End of video/livestream")
            break
        cv.imshow("Ball Tracking", track_ball(frame))
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    
cv.destroyAllWindows()