import cv2 as cv
import os

videos = []
num = 1

for file in os.listdir("videos"):
    if file.endswith((".mp4", ".m4a", ".MOV")):
        videos.append(file)
        print(f"{num}. {file}")
        num+=1
index = int(input("Select a video by typing the file number: "))
video = cv.VideoCapture(f"videos/{videos[index-1]}")

total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))


count, success = 0, True

while success:
    success, image = video.read() # Read frame
    if success: 
        cv.imwrite(f"frames/frame{count}.jpg", image) # Save frame
        count += 1
        print(f'{video.get(cv.CAP_PROP_POS_FRAMES)}/{total_frames}')
        

video.release()