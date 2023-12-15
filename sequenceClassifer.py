import json
import cv2

def calculate_center(box):
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return (x_center, y_center)


video_path = 'videos/avi/01_2023-08-02_18-29-59.avi'
json_seq_path = video_path.replace('.avi', '_seq.json')

with open(json_seq_path, 'r') as file:
    sequences = json.load(file)

cap = cv2.VideoCapture(video_path)  # Replace with your video path
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sequence_types = {ord('1'): 'guard', ord('2'): 'exit', ord('3'): 'enter', ord('4'): 'other'}

sequence_id = 0

while sequence_id < len(sequences):
    seq = sequences[sequence_id]
    paths = []

    if not seq['data']:  # Check if data is empty
        sequence_id += 1
        continue  # Skip this sequence if there are no frames

    for frame_data in seq['data']:
        for box in frame_data['boxes']:
            paths.append(calculate_center(box[:4]))

    if not paths:  # Skip if there are no paths
        sequence_id += 1
        continue    

    middle_idx = len(seq['data']) // 2
    middle_frame = seq['data'][middle_idx]['frame']
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    if not ret:
        break
    for i in range(1, len(paths)):
        pt1 = (int(paths[i - 1][0]), int(paths[i - 1][1]))
        pt2 = (int(paths[i][0]), int(paths[i][1]))
        cv2.line(frame, pt1, pt2, (255, 0, 0), 1)

    first_point = (int(paths[0][0]), int(paths[0][1]))
    last_point = (int(paths[-1][0]), int(paths[-1][1]))
    cv2.circle(frame, first_point, 5, (0, 255, 0), -1)  # Green circle for first point
    cv2.circle(frame, last_point, 5, (0, 0, 255), -1)   # Red circle for last point

    sequence_info = f'Sequence ID: {sequence_id}/{len(sequences)}, Length: {seq["sequence_length"]}, Type:{seq["type"]}'
    text_size, _ = cv2.getTextSize(sequence_info, cv2.FONT_HERSHEY_PLAIN, 1, 1)
    text_x = frame.shape[1] - 10 - text_size[0]
    cv2.putText(frame, sequence_info, (text_x, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    text_help = "1: guard, 2: exit, 3: enter, 4: other"
    cv2.putText(frame, text_help, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)


    cv2.imshow('Sequence Visualization', frame)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    elif key == ord('b'):  # 'b' key for going back
        sequence_id = max(0, sequence_id - 1)  # Ensure it doesn't go below 0
        print(sequence_id)
        continue
    elif key in sequence_types:
        seq['type'] = sequence_types[key]  # Update the sequence type
        with open(json_seq_path, 'w') as outfile:
            json.dump(sequences, outfile, indent=4)
        sequence_id += 1
    else:
        sequence_id += 1  
            

# Save the updated sequences back to the JSON file
with open(json_seq_path, 'w') as outfile:
    json.dump(sequences, outfile, indent=4)

cap.release()
cv2.destroyAllWindows()
