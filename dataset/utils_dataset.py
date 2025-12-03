import os
import numpy as np

def get_labels_for_npz(npz_file, annotations):
    # es: "10_3_360.mp4_1s_1s.npz"
    base = os.path.basename(npz_file)
    activity, attempt = base.split("_")[:2]  # "10", "3"
    recording_id = f"{activity}_{attempt}"

    # carica feature
    data = np.load(npz_file)
    arr = data[list(data.keys())[0]]  # shape (N, 1024)
    N = arr.shape[0]

    labels = np.zeros(N, dtype=np.int64)  # default: normal = no-error = 0

    # trova annotation di questo recording
    info = annotations[recording_id]
    steps = info["steps"]

    # assegnazione label per ogni secondo
    for step in steps:
        has_error = int(step["has_errors"])  # True→1, False→0
        start = step["start_time"]
        end   = step["end_time"]

        if start == -1 or end == -1 or has_error == 0:
            continue

        for sec in range(int(start), int(end) + 1, 1):
            sec_start = sec
            sec_end   = sec + 1

            # check overlap
            if sec_start >= start and sec_end <= end: # i secondi ai bordi avranno sempre il valore di default (norml = no-error = 0)
                labels[sec] = has_error

    return arr, labels