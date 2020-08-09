import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import torch
from waymo_open_dataset.utils import frame_utils, box_utils, transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from utils import label_to_box
from Tracker import Tracker

print(f"TensorFlow version: {tf.__version__}")
print(f"Torch version: {torch.__version__}")

def main():
    segment_path = "/home/erik/Projects/notebooks/pointclouds/data/segment-15578655130939579324_620_000_640_000_with_camera_labels.tfrecord"

    dataset = tf.data.TFRecordDataset([segment_path])
    frames = []

    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frames.append(frame)

    print(f"Loaded {len(frames)} frames.")

    tracking_id_to_idx = {}
    tracker = Tracker()

    for i, frame in enumerate(frames):
        print(f"Frame {i}")
        labels = frame.laser_labels

        for label in labels:
            if label.id not in tracking_id_to_idx:
                tracking_id_to_idx[label.id] = 150 + len(tracking_id_to_idx)

        detections = np.array([label_to_box(label, tracking_id_to_idx) for label in labels])
        print(f"{len(detections)} detections.")
        tracker.update(detections)
        print()

    """
    all_accuracies = []
    label_accuracies = { 1: [], 2: [], 3: [], 4: [] }

    matched = (predicted_id == boxes[:,8]).sum()
    accuracy = matched / len(predicted_id) 
    all_accuracies.append(accuracy)

    out = f"{accuracy:.2f} ({matched} / {len(predicted_id)})"
    print(f"{out:<20}", end='')
    #print(f"\t{'Total':<13}{accuracy:.2f} ({matched} / {len(predicted_id)})")

    for t in label_to_str.keys():
        indices = np.where(boxes[:,7] == t)[0]
        if len(indices) > 0:
            matched = (predicted_id[indices] == boxes[indices][:,8]).sum()
            accuracy = matched / len(indices)
            out = f"{accuracy:.2f} ({matched} / {len(indices)})"
            print(f"{out:<20}", end='')

            label_accuracies[t].append(accuracy)

            #print(f"\t{label_to_str[t]:<13}")
        else:
            print(f"{'-':<20}", end='')
            #print(f"\t{label_to_str[t]:<13}-")
    """

if __name__ == '__main__':
    main()
