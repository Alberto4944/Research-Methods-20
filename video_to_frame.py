import cv2 as cv
import os
from pathlib import Path
import math

for file in os.listdir("videos_for_labelling"):
    if file.endswith((".mp4", ".m4a", ".MOV", ".mov")):
        video = cv.VideoCapture(f"videos_for_labelling/{file}")
        total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))


        count, success = 0, True
        frame_interval = math.floor(total_frames/25);

        while success:
            success, image = video.read() # Read frame
            if success and count % frame_interval == 0: 
                cv.imwrite(f"frames/{Path(file).stem}_FRAME_{count}.jpg", image) # Save frame
                print(f'{file} - {video.get(cv.CAP_PROP_POS_FRAMES)}/{total_frames}')   
            count += 1     

        video.release()
    