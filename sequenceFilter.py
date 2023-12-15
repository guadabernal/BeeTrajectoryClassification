import json
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import time

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_center(box):
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return (x_center, y_center)

def pathTracking(sequence):
    bees_tracks = {}
    next_bee_id = 0
    max_distance = 150

    for frame in sequence['data']:
        current_boxes = [calculate_center(box) for box in frame['boxes']]
        current_assignments = {}

        if not bees_tracks:
            for box_center in current_boxes:
                bees_tracks[next_bee_id] = [box_center]
                next_bee_id += 1
        else:
            last_bee_ids = list(bees_tracks.keys())
            last_bee_centers = [bees_tracks[bee_id][-1] for bee_id in last_bee_ids]
            distance_matrix = cdist(last_bee_centers, current_boxes)
            for i, last_center in enumerate(last_bee_centers):
                distances_to_current = distance_matrix[i]
                closest_current_idx = np.argmin(distances_to_current)
                distance_to_closest = distances_to_current[closest_current_idx]

                if distance_to_closest < max_distance:
                    bee_id = last_bee_ids[i]
                    bees_tracks[bee_id].append(current_boxes[closest_current_idx])
                    current_assignments[bee_id] = closest_current_idx

            for i, box_center in enumerate(current_boxes):
                if i not in current_assignments.values():
                    bees_tracks[next_bee_id] = [box_center]
                    next_bee_id += 1

    return bees_tracks

video_path = 'videos/avi/01_2023-08-02_18-29-59.avi'
json_seq_path = video_path.replace('.avi', '_seq.json')

with open(json_seq_path, 'r') as file:
    sequences = json.load(file)

enter_sequences = sequences
#[seq for seq in sequences if seq['type'] == 'guard']

cap = cv2.VideoCapture(video_path)


colors = [
    (180, 0, 0),  # Red
    (0, 128, 0),  # Green
    (0, 0, 180),  # Blue
    (180, 150, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255)  # Yellow
]
id = 0
while 0 <= id < len(enter_sequences):
    seq = enter_sequences[id]
    bees_tracks = pathTracking(seq)
    frames = [frame_data['frame'] for frame_data in seq['data']]

    playing = False
    frame_idx = 0
    video_writer = None

    while True:  # for playing and pausing
        if playing:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[frame_idx])
            ret, frame = cap.read()
            frame_idx = (frame_idx + 1) % len(frames)
        else:
            # Set to the first frame only when not in playing mode
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
            ret, frame = cap.read()

        if not ret:
            break

        bees_tracks = pathTracking(seq)
        for bee_id, path in bees_tracks.items():
            bee_color = colors[bee_id % len(colors)]

            for i in range(1, len(path)):
                pt1 = (int(path[i - 1][0]), int(path[i - 1][1]))
                pt2 = (int(path[i][0]), int(path[i][1]))
                cv2.line(frame, pt1, pt2, bee_color, 2, cv2.LINE_AA)

            # Draw the starting and ending points with a colored border
            if path:
                first_point = (int(path[0][0]), int(path[0][1]))
                last_point = (int(path[-1][0]), int(path[-1][1]))

                # Draw the starting point (green with a colored border)
                cv2.circle(frame, first_point, 7, bee_color, -1)  # Colored border
                cv2.circle(frame, first_point, 5, (0, 255, 0), -1)  # Green for the start

                # Draw the ending point (red with a colored border)
                cv2.circle(frame, last_point, 7, bee_color, -1)  # Colored border
                cv2.circle(frame, last_point, 5, (0, 0, 255), -1)  # Red for the end

        sequence_info = f'ID: {id}/{len(enter_sequences)}, SeqId: {seq["sequence_id"]}, Length: {len(seq["data"])}, Type: {seq["type"]}'
        cv2.putText(frame, sequence_info, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        cv2.imshow('Sequence Visualization', frame)

        if video_writer is not None:
            video_writer.write(frame)

        key = cv2.waitKey(1 if playing else 0)  # Wait only 1 ms if playing

        if key == ord('q'):
            break
        elif key == ord('b') and not playing:  # 'b' key for going back
            id = max(0, id - 1)
            break
        elif key == ord('n') and not playing:  # 'n' key for next sequence
            id += 1
            break
        elif key == ord('p'):  # 'p' key to play/pause the sequence
            playing = not playing  # Toggle the play state
        elif key == ord('d'): # dump frames and boxes
            for s in seq['data']:
                print(s['frame'], s['boxes'])
        elif key == ord('s') and not playing:  # 's' key to save the sequence
            if video_writer is None:
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_size = (frame.shape[1], frame.shape[0])
                video_writer = cv2.VideoWriter('output.avi', fourcc, fps, frame_size)
            else:
                # Save and close the video writer and reset it for next use
                video_writer.release()
                video_writer = None
        if playing:
            frame_idx = (frame_idx + 1) % len(frames)  # Loop back to the start

    if key == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
