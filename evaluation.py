import torch
import numpy as np
from iou3d import iou3d
from scipy.optimize import linear_sum_assignment

def evaluation(real_objects, hypothesis, n_frames):
    mapping = {}
    distances = []
    c = []
    fp = []
    misses = []
    mme = []

    def count_in_frames(target):
        return np.array([[1 if frame in entries else 0 for frame in range(n_frames)] for entries in target.values()]).sum(axis=0)

    n_o = count_in_frames(real_objects)
    n_h = count_in_frames(hypothesis)

    for frame in range(n_frames):
        print(f"Evaluating frame {frame} / {n_frames}")

        frame_mme = 0
        #distances.append([])
        frame_mapping = {}

        o_candidates = []
        o_indices = []
        for o_idx in real_objects.keys():
            if frame not in real_objects[o_idx]: continue
            o_candidates.append(real_objects[o_idx][frame])
            o_indices.append(o_idx)

        h_candidates = []
        h_indices = []
        for h_idx in hypothesis.keys():
            if frame not in hypothesis[h_idx]: continue
            h_candidates.append(hypothesis[h_idx][frame])
            h_indices.append(h_idx)

        o_candidates = torch.Tensor(o_candidates)
        h_candidates = torch.Tensor(h_candidates)

        overlap = iou3d(o_candidates, h_candidates)

        for o_i, o_idx in enumerate(o_indices):
            ious, indices = overlap[o_i].sort(descending=True)

            for i in range(len(overlap[o_i])):
                if ious[i].item() == 0: break # No overlap

                h_idx = h_indices[indices[i]]

                if h_idx in frame_mapping.values(): # h already used
                    continue

                frame_mapping[o_idx] = h_idx
                break

        """
        row_ind, col_ind = linear_sum_assignment(overlap, maximize=True)
        matched_indices = np.stack((row_ind, col_ind), axis=1)
        for o_i, h_i in matched_indices:
            frame_mapping[o_indices[o_i]] = h_indices[h_i] 
            
        for o_idx in o_candidates: 
            candidates = []
            indices = []
            for h_idx in hypothesis.keys():
                if h_idx in frame_mapping.values():
                    continue
                if frame in hypothesis[h_idx]:
                    candidates.append(hypothesis[h_idx][frame])
                    indices.append(h_idx)

            candidates = torch.Tensor(candidates)
            target = torch.Tensor(real_objects[o_idx][frame]).reshape((1, 8))
            best = iou3d(target, candidates)[0].argmax().item()
            frame_mapping[o_idx] = indices[best]
        """

        for o_idx, h_idx in frame_mapping.items():
            #d = iou3d(
            #    torch.Tensor(real_objects[o_idx][frame]).reshape((1, 8)),
            #    torch.Tensor(hypothesis[h_idx][frame]).reshape((1, 8))
            #)[0][0]
            #distances[frame].append(d)

            if frame > 0 and o_idx in mapping:
                if mapping[o_idx] != h_idx:
                    frame_mme += 1

        c.append(len(frame_mapping))
        fp.append(n_h[frame] - len(frame_mapping))
        misses.append(n_o[frame]- len(frame_mapping))
        mme.append(frame_mme)

        mapping = frame_mapping

    #distance_sum = np.array([d for frame_distances in distances for d in frame_distances]).sum()
    #c_sum = np.array(c).sum()
    #MOTP = distance_sum / c_sum
    #print(f"MOTP: {MOTP}")

    g = n_o
    g_sum = np.array(g).sum()
    misses_sum = np.array(misses).sum()
    fp_sum = np.array(fp).sum()
    mme_sum = np.array(mme).sum()

    MOTA = 1 - (misses_sum + fp_sum + mme_sum) / g_sum
    print(f"MOTA: {MOTA}")
