import torch
import numpy as np
from iou3d import iou3d
from utils import labels, label_to_str
from scipy.optimize import linear_sum_assignment

def get_type(target):
    return target[list(target.keys())[0]][7]

def evaluation(real_objects, hypothesis, n_frames):
    print("Running evaluation.")

    print(f"Number of real objects: {len(real_objects)}")
    print(f"Number of predicted objects: {len(hypothesis)}")

    mapping = {}
    
    c = []
    fp = []
    misses = []
    mme = []

    c_l = { label: [] for label in labels }
    fp_l = { label: [] for label in labels }
    misses_l = { label: [] for label in labels }
    mme_l = { label: [] for label in labels }
    n_o_l = { label: [] for label in labels }
    n_h_l = { label: [] for label in labels }

    def count_in_frames(target):
        return np.array([[1 if frame in entries else 0 for frame in range(n_frames)] for entries in target.values()]).sum(axis=0)

    n_o = count_in_frames(real_objects)
    n_h = count_in_frames(hypothesis)

    for frame in range(n_frames):
        # print(f"Evaluating frame {frame} / {n_frames}")

        frame_mme = 0
        frame_l_mme = { label: 0 for label in labels }
        #distances.append([])
        frame_mapping = {}
        n_o_l_frame = { label: 0 for label in labels }
        n_h_l_frame = { label: 0 for label in labels }

        o_candidates = []
        o_indices = []
        for o_idx in real_objects.keys():
            if frame not in real_objects[o_idx]: continue
            o_candidates.append(real_objects[o_idx][frame])
            o_indices.append(o_idx)
            n_o_l_frame[get_type(real_objects[o_idx])] += 1

        h_candidates = []
        h_indices = []
        for h_idx in hypothesis.keys():
            if frame not in hypothesis[h_idx]: continue
            h_candidates.append(hypothesis[h_idx][frame])
            h_indices.append(h_idx)
            n_h_l_frame[get_type(hypothesis[h_idx])] += 1

        o_candidates = torch.Tensor(o_candidates)
        h_candidates = torch.Tensor(h_candidates)

        overlap = iou3d(o_candidates, h_candidates)

        # set overlap of boxes with different types to 0
        for d in range(overlap.shape[0]):
            for p in range(overlap.shape[1]):
                if o_candidates[d,7] != h_candidates[p,7]:
                    overlap[d,p] = 0
                    
        frame_mapping = mapping_from_overlap(overlap, o_indices, h_indices)

        for o_idx, h_idx in frame_mapping.items():
            if frame > 0 and o_idx in mapping:
                if mapping[o_idx] != h_idx:
                    frame_mme += 1
                    frame_l_mme[get_type(real_objects[o_idx])] += 1

        c.append(len(frame_mapping))
        fp.append(n_h[frame] - len(frame_mapping))
        misses.append(n_o[frame]- len(frame_mapping))
        mme.append(frame_mme)

        c_l_frame = { label: 0 for label in labels }

        for o_idx, h_idx in frame_mapping.items():
            assert get_type(real_objects[o_idx]) == get_type(hypothesis[h_idx])
            label = get_type(real_objects[o_idx])
            c_l_frame[label] += 1

            if n_h_l_frame[label] < c_l_frame[label]:
                print(f"Something went wrong: having {c_l_frame[label]} mapping entries for label {label_to_str[label]} but only {n_h_l_frame[label]} hypothesis")
                return

            if n_o_l_frame[label] < c_l_frame[label]:
                print(f"Something went wrong: having {c_l_frame[label]} mapping entries for label {label_to_str[label]} but only {n_o_l_frame[label]} true objects")
                return

        for label in labels:
            c_l[label].append(c_l_frame[label])
            fp_l[label].append(n_h_l_frame[label] - c_l_frame[label])
            misses_l[label].append(n_o_l_frame[label] - c_l_frame[label])
            mme_l[label].append(frame_l_mme[label])
            n_h_l[label].append(n_h_l_frame[label])
            n_o_l[label].append(n_o_l_frame[label])

        mapping = frame_mapping

    g = n_o
    g_sum = np.array(g).sum()
    c_sum = np.array(c).sum()
    misses_sum = np.array(misses).sum()
    fp_sum = np.array(fp).sum()
    mme_sum = np.array(mme).sum()

    for frame in range(n_frames):
        frame_str = f"[Frame {frame}]"
        print(f"{frame_str:12} g {g[frame]} | misses {misses[frame]} | fp {fp[frame]} | mme {mme[frame]}")

    print(f"{'Total':12} g {g_sum} | misses {misses_sum} | fp {fp_sum} | mme {mme_sum}")

    MOTA = 1 - (misses_sum + fp_sum + mme_sum) / g_sum
    print(f"{'All':10} MOTA: {MOTA} | c {c_sum} | g {g_sum} | misses {misses_sum} | fp {fp_sum} | mme {mme_sum}")

    for label in labels:
        g_sum = np.array(n_o_l[label]).sum()
        if g_sum == 0: continue

        c_sum = np.array(c_l[label]).sum()
        misses_sum = np.array(misses_l[label]).sum()
        fp_sum = np.array(fp_l[label]).sum()
        mme_sum = np.array(mme_l[label]).sum()

        MOTA = 1 - (misses_sum + fp_sum + mme_sum) / g_sum
        print(f"{label_to_str[label]:10} MOTA: {MOTA} | c {c_sum} | g {g_sum} | misses {misses_sum} | fp {fp_sum} | mme {mme_sum}")


def mapping_from_overlap(overlap, o_indices, h_indices):
    threshold = 0
    mapping = {}

    for o_i, o_idx in enumerate(o_indices):
        ious, indices = overlap[o_i].sort(descending=True)

        for i in range(len(overlap[o_i])):
            if ious[i].item() <= threshold: break # No enough overlap

            h_idx = h_indices[indices[i]]

            if h_idx in mapping.values(): # h already used
                continue

            mapping[o_idx] = h_idx
            break

    return mapping
