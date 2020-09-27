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

print(f"TensorFlow version: {tf.__version__}")
print(f"Torch version: {torch.__version__}", end="\n\n")

s_pedestrians = 'segment-12956664801249730713_2840_000_2860_000_with_camera_labels.tfrecord'
s_vehicles = 'segment-15578655130939579324_620_000_640_000_with_camera_labels.tfrecord'

def print_chart(objects, n):
    for id, entries in objects.items():
        type_str = label_to_str[entries[list(entries.keys())[0]][7]]
        r = range(0, n)
        print(f"{id:3} {type_str:10} {''.join(['▮' if i in entries else '▯' for i in r])}")

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
    tracker = NaiveTracker()
    real_objects = {}
    hypothesis = {}

    for i, frame in enumerate(frames):
        print(f"Frame {i} / {len(frames)}")
        labels = frame.laser_labels

        labels = [label for label in labels if label.type != 3]

        for label in labels:
            if label.id not in label_id_to_idx:
                label_id_to_idx[label.id] = len(label_id_to_idx)

        for label in labels: 
            idx = label_id_to_idx[label.id]
            if idx not in real_objects:
                real_objects[idx] = {}
            real_objects[idx][i] = label_to_box(label)

        detections = np.array([label_to_box(label) for label in labels])

        print(f"{len(detections)} detections.")

        frame_hypothesis = tracker.update(detections)
        for h_idx in frame_hypothesis.keys():
            if h_idx not in hypothesis:
                hypothesis[h_idx] = {}
            hypothesis[h_idx][i] = frame_hypothesis[h_idx]

    print("Real objects")
    print_chart(real_objects, len(frames))
    print()

    print("Hypothesis")
    print_chart(hypothesis, len(frames))

    evaluation(real_objects, hypothesis, len(frames))

if __name__ == '__main__':
    main()
