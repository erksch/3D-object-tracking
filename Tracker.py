import tensorflow as tf
import torch
import numpy as np
from iou3d import iou3d
from scipy.optimize import linear_sum_assignment
from utils import label_to_str, label_to_box
from KalmanTracker import KalmanTracker

class Tracker:
    def __init__(self):
        self.trackers = []
        self.id_count = 0

    def update(self, detections):
        matched, unmatched_detections, unmatched_trackers = assign_detections_to_trackers(detections, self.trackers)

        print(f"Matched {len(matched)} of {len(self.trackers)} trackers and {len(detections)} detections.")

        for t, tracker in enumerate(self.trackers):
            if t not in unmatched_trackers:
                d = matched[np.where(matched[:,1] == t)[0], 0][0]
                tracker.update(detections[d,:7])

        print(f"Creating {len(unmatched_detections)} trackers for unmatched detections.")
        print("Unmatched: ", end='')
        for i in unmatched_detections:
            type = detections[i][7]
            print(f"{i} ({label_to_str[type]})", end=', ')
            tracker = KalmanTracker(detections[i,:7], self.id_count, type)
            self.id_count += 1
            self.trackers.append(tracker)
        print()

        print(f"Removing {len(unmatched_trackers)} unmatched trackers.")
        for i in unmatched_trackers:
            self.trackers.pop(i)

        hypothesis = {}

        for tracker in self.trackers:
            hypothesis[tracker.id] = np.append(tracker.get_state(), tracker.type)

        return hypothesis

def assign_detections_to_trackers(detections, trackers, iou_threshold=0.1):
    """
	Assigns detections to tracked object (both represented as bounding boxes)
	detections:  N x 8 x 3
	trackers:    M x 8 x 3
	Returns 3 lists of matches, unmatched_detections and unmatched_trackers
	"""
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 8, 3), dtype=int) 

    predictions = np.zeros((len(trackers), 7))

    for t, tracker in enumerate(trackers):
        tracker.predict()
        predictions[t] = tracker.get_state()

    cost_matrix = iou3d(
        torch.Tensor(detections)[:,:7],
        torch.Tensor(predictions)[:,:7]
    ).numpy()

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    matched_indices = np.stack((row_ind, col_ind), axis=1)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]): 
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]): unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if cost_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
