import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import torch
from waymo_open_dataset.utils import frame_utils, box_utils, transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from utils import label_to_box, label_to_str
from evaluation import evaluation
from Tracker import Tracker
from NaiveTracker import NaiveTracker

np.random.seed(0)

print(f"TensorFlow version: {tf.__version__}")
print(f"Torch version: {torch.__version__}", end="\n\n")

s_pedestrians = 'segment-12956664801249730713_2840_000_2860_000_with_camera_labels.tfrecord'
s_vehicles = 'segment-15578655130939579324_620_000_640_000_with_camera_labels.tfrecord'

def print_chart(objects, n):
    for label in ['vehicle', 'pedestrian']:
        c = 0
        for id in reversed(list(objects.keys())):
            entries = objects[id]
            type_str = label_to_str[entries[list(entries.keys())[0]][7]]
            if type_str != label: continue
            c += 1
            r = range(0, n)
            COLOR = '\033[91m' if type_str == 'vehicle' else '\033[94m'
            ENDC = '\033[0m'
            print(f"{COLOR}{id:3} {type_str:10} {''.join(['▮' if i in entries else '-' for i in r])}{ENDC}")
        print(f"Label {c}")

def print_mappings(real_objects, hypothesis, mappings, n):
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    frame_range = range(0, n)
    mme = {}
    misses = {}
    false_positives = {}
    mismatch_memory = {}


    for label in ['vehicle', 'pedestrian']:
        mme[label] = 0
        misses[label] = 0
        false_positives[label] = 0

        for o_idx in reversed(list(real_objects.keys())):
            mismatch_memory[o_idx] = []
            entries = real_objects[o_idx]
            type_str = label_to_str[entries[list(entries.keys())[0]][7]]
            if type_str != label: continue
            print(f"{o_idx:3} {type_str:10}", end='')
            for frame in frame_range:
                if frame not in entries:
                    print(f"-", end='')
                elif o_idx not in mappings[frame]:
                    print(f"{RED}▮{ENDC}", end='')
                    misses[label] += 1
                elif frame > 0 :
                    found_mismatch = False
                    for prev_frame in reversed(range(frame)):
                        if prev_frame in mismatch_memory[o_idx]: break
                        if o_idx in mappings[prev_frame] and mappings[prev_frame][o_idx] != mappings[frame][o_idx]:
                            print(f"{BLUE}▮{ENDC}", end='')
                            mme[label] += 1
                            found_mismatch = True
                            mismatch_memory[o_idx].append(prev_frame)
                            break
                    if not found_mismatch:
                        print(f'▯', end='')
                else:
                    print(f'▯', end='')
            print()

    print('Misses')
    print(misses)
    print('mme')
    print(mme)
    print('false_positives')
    print(false_positives)

def main():
    segment_path = f"/home/erik/Projects/notebooks/pointclouds/data/{s_pedestrians}"

    dataset = tf.data.TFRecordDataset([segment_path])
    frames = []

    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frames.append(frame)

    print(f"Loaded {len(frames)} frames.", end="\n\n")

    label_id_to_idx = {}
    real_objects = {}
    real_objects_to_label = {}
    hypothesis = {}

    object_counts = {}

    tracker = Tracker()
    use_dropout = False
    dropout = 0.2

    for i, frame in enumerate(frames):
        print(f"Frame {i} / {len(frames)}")
        labels = frame.laser_labels

        labels = [label for label in labels if label.type != 3]

        for label in labels:
            if label.id not in label_id_to_idx:
                label_id_to_idx[label.id] = len(label_id_to_idx)

        detections = np.array([label_to_box(label, label_id_to_idx[label.id]) for label in labels])
        print(f"{len(detections)} detections.")

        if use_dropout:
            n_detections = detections.shape[0]
            n_dropout = int(n_detections * dropout)
            print(f"Randomly dropping out {n_dropout} detections.")
            detections = detections[np.random.choice(n_detections, n_detections - n_dropout, replace=False)]


        for detection in detections: 
            type = int(detection[7])
            if type not in object_counts:
                object_counts[type] = 1
            else: 
                object_counts[type] += 1

            idx = int(detection[8])
            if idx not in real_objects:
                real_objects[idx] = {}
            real_objects[idx][i] = detection[:8]

        frame_hypothesis = tracker.update(detections[:,:8])
        for h_idx in frame_hypothesis.keys():
            if h_idx not in hypothesis:
                hypothesis[h_idx] = {}
            hypothesis[h_idx][i] = frame_hypothesis[h_idx]

    print(object_counts)

    print("Real objects")
    print_chart(real_objects, len(frames))
    print()

    print("Hypothesis")
    print_chart(hypothesis, len(frames))

    mappings = evaluation(real_objects, hypothesis, len(frames))

    print_mappings(real_objects, hypothesis, mappings, len(frames))

if __name__ == '__main__':
    main()
