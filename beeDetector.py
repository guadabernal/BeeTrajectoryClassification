from ultralytics import YOLO
import cv2
from tqdm import tqdm
import json
import os
import torch

# Video path
video_path = 'videos_avi/01_2023-08-02_21-17-53.avi'

# Model path
pt_path = "best.pt"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (GPU support).")
else:
    device = torch.device("cpu")
    print("Using CPU.")
    
def beeDetection(model, frame):
    results = model.predict(frame, device, verbose=False)
    returnBoxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            b = box.xyxy[0].tolist()    # get box coordinates in (top, left, bottom, right) format
            c = box.conf.item() * 100   # get the confidence in %
            returnBoxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3]), c])
    return returnBoxes        


yolo = YOLO(pt_path)

json_file_path = os.path.splitext(video_path)[0] + '.json'

try:
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    sorted_data = sorted(data, key=lambda x: x['frame'])
    with open(json_file_path, 'w') as file:
        json.dump(sorted_data, file)
    with open(json_file_path, 'r') as file:
        detections = json.load(file)
        start_frame = detections[-1]['frame'] + 1 if detections else 0
except (FileNotFoundError, json.JSONDecodeError):
    start_frame = 0
    detections = []

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

pbar = tqdm(initial=start_frame, total=total_frames, desc="Processing Video")

update_interval = 1000

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        iFrame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        boxes = beeDetection(yolo, frame)
        if boxes:
            detections.append({"frame": iFrame, "boxes": boxes})
        pbar.update(1)

        if iFrame % update_interval == 0:
            with open(json_file_path, 'w') as file:
                json.dump(detections, file)
    else:
        break

# update the JSON file at the end of processing
with open(json_file_path, 'w') as file:
    json.dump(detections, file)

cap.release()