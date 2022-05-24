import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import torch
import argparse
from waymo_open_dataset.utils import frame_utils, box_utils, transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from utils import label_to_box, label_to_str, print_chart, print_mappings
from evaluation import evaluation
from PredictiveTracker import PredictiveTracker
from NaiveTracker import NaiveTracker

np.random.seed(0)

def main(arguments):
    print("============================")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Torch version: {torch.__version__}")
    print("============================", end="\n\n")

    print(f"Loading segment from path {arguments.segment}")

    dataset = tf.data.TFRecordDataset([arguments.segment])
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

    if arguments.tracker == 'predictive':
        tracker = PredictiveTracker()
    else:
        tracker = NaiveTracker()

    for i, frame in enumerate(frames):
        print(f"Frame {i} / {len(frames)}")
        labels = frame.laser_labels

        labels = [label for label in labels if label.type != 3]

        for label in labels:
            if label.id not in label_id_to_idx:
                label_id_to_idx[label.id] = len(label_id_to_idx)

        detections = np.array([label_to_box(label, label_id_to_idx[label.id]) for label in labels])
        print(f"{len(detections)} detections.")

        if arguments.dropout > 0:
            n_detections = detections.shape[0]
            n_dropout = int(n_detections * arguments.dropout)
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
    parser = argparse.ArgumentParser(description='Run a tracking algorithmn on a segment of the Waymo Open Dataset')

    parser.add_argument('-s', '--segment', type=str, required=True,
                        help='Path to a segment .tfrecord file.')
    parser.add_argument('-d', '--dropout', type=float, default=0,
                        help='Dropout rate for detections in each frame. Default is 0.')
    parser.add_argument('-t', '--tracker', type=str, default="predictive",
                        help='Tracker to use. Either predictive or naive. Default is predictive.')

    arguments = parser.parse_args()
    main(arguments)
