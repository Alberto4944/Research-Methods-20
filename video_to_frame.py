import cv2 as cv
import os
from pathlib import Path
import math

videos = []
num = 1

for file in os.listdir("videos"):
    if file.endswith((".mp4", ".m4a", ".MOV", ".mov")):
        videos.append(file)
        print(f"{num}. {file}")
        num+=1
index = int(input("Select a video by typing the file number: "))
video = cv.VideoCapture(f"videos/{videos[index-1]}")

total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))


count, success = 0, True
frame_interval = math.floor(total_frames/25);
video_name = Path(f"videos/{videos[index-1]}").stem

while success:
    success, image = video.read() # Read frame
    if success and count % frame_interval == 0: 
        cv.imwrite(f"frames/{video_name}_FRAME_{count}.jpg", image) # Save frame
        print(f'{video.get(cv.CAP_PROP_POS_FRAMES)}/{total_frames}')   
    count += 1     

video.release()