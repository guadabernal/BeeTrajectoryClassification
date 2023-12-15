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

def beeDetection(model, frames, device):
    results = model.predict(frames, device, verbose=False)
    batch_detections = []
    for frame_results in results:
        frame_boxes = []
        for result in frame_results:
            boxes = result.boxes
            for box in boxes:
                b = box.xyxy[0].tolist()
                c = box.conf.item() * 100
                frame_boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3]), c])
        batch_detections.append(frame_boxes)
    return batch_detections

# Initialize YOLO model
yolo = YOLO(pt_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Video handling
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Load previous detections or start anew
json_file_path = os.path.splitext(video_path)[0] + '.json'
try:
    with open(json_file_path, 'r') as file:
        detections = json.load(file)
    start_frame = detections[-1]['frame'] + 1 if detections else 0
except (FileNotFoundError, json.JSONDecodeError):
    start_frame = 0
    detections = []

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(initial=start_frame, total=total_frames, desc="Processing Video")
update_interval = 1000
batch_size = 8  # Adjust based on your GPU capacity

batch_frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        batch_frames.append(frame)
        if len(batch_frames) == batch_size:
            boxes_batch = beeDetection(yolo, batch_frames, device)
            for idx, boxes in enumerate(boxes_batch):
                iFrame = start_frame + idx
                if boxes:
                    detections.append({"frame": iFrame, "boxes": boxes})
            start_frame += batch_size
            batch_frames = []
            pbar.update(batch_size)
    else:
        break

# Process remaining frames
if batch_frames:
    boxes_batch = beeDetection(yolo, batch_frames, device)
    for idx, boxes in enumerate(boxes_batch):
        iFrame = start_frame + idx
        if boxes:
            detections.append({"frame": iFrame, "boxes": boxes})
    pbar.update(len(batch_frames))

# Update JSON file at the end
with open(json_file_path, 'w') as file:
    json.dump(detections, file)

cap.release()
