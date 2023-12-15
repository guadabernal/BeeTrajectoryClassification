import json

# Select video json file
json_file_path = 'videos_avi/01_2023-08-02_18-29-59.json'

def calculate_center(box):
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return (x_center, y_center)

def calculate_iou(boxA, boxB):
    # determine coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def filter_boxes(boxes, confidence_threshold = 60.0, iou_threshold = 0.1):
    filtered_boxes = [box for box in boxes if box[4] >= confidence_threshold]
    filtered_boxes.sort(key=lambda x: x[4], reverse=True)

    new_boxes = []
    while filtered_boxes:
        boxA = filtered_boxes.pop(0)  # take the first box (highest confidence)
        new_boxes.append(boxA)  # assume this box is to be kept
        filtered_boxes = [boxB for boxB in filtered_boxes if calculate_iou(boxA[:4], boxB[:4]) <= iou_threshold]

    return new_boxes

# load data from JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

sequence_gap_max = 10
min_sequence_len = 10
sequences = []
current_sequence = {'sequence_id':0, 'type': 'unknown', 'sequence_length':0, 'data': []}

confidence_threshold = 50.0     # set confidence threshold
iou_threshold = 0.1             # set intersection of union threshold

last_frame = 0
seqId = 0
for i in range(0, len(data)):
    fboxes = filter_boxes(data[i]['boxes'], confidence_threshold, iou_threshold)
    if not fboxes:
        continue
    current_frame = data[i]['frame']
    if current_sequence['data'] == [] or current_frame - last_frame <= sequence_gap_max:        
        current_sequence['data'].append({"frame":current_frame, "boxes": fboxes})
        last_frame = current_frame
    else:
        sequence_len = len(current_sequence['data'])
        if sequence_len > min_sequence_len:
            current_sequence['sequence_length'] = sequence_len
            sequences.append(current_sequence)        
            seqId += 1
        current_sequence = {'sequence_id':seqId, 'type': 'unknown', 'sequence_length':0, 'data': []}

# add last sequence if it's not empty
sequence_len = len(current_sequence['data'])
if  sequence_len > min_sequence_len:
    current_sequence['sequence_length'] = sequence_len
    sequences.append(current_sequence)

json_seq_path = json_file_path.replace('.json', '_seq.json')
with open(json_seq_path, 'w') as outfile:
    json.dump(sequences, outfile, indent=4)

ranges = {
    "0-10": 0,
    "10-20": 0,
    "20-30": 0,
    "30-40": 0,
    "40+": 0
}

print("Number of sequences = ", len(sequences))

# Count the sequences
for sequence in sequences:
    length = sequence['sequence_length']
    if 0 <= length <= 10:
        ranges["0-10"] += 1
    elif 10 < length <= 20:
        ranges["10-20"] += 1
    elif 20 < length <= 30:
        ranges["20-30"] += 1
    elif 30 < length <= 40:
        ranges["30-40"] += 1
    else:
        ranges["40+"] += 1

# print the counts
for range, count in ranges.items():
    print(f"Number of sequences with {range} frames: {count}")
