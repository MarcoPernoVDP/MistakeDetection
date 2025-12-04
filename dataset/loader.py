import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class CaptainCook4D_Dataset(Dataset):
    """
    Dataset per CaptainCook4D basato su file .npz (es. feature Omnivore)
    e annotazioni JSON (complete_step_annotations.json).
    """
    
    def __init__(self, features_dir, annotations_dir):
        """
        Args:
            omnivore_dir (str): path alla cartella con i file .npz
            annotations_dir (str): path alla cartella con complete_step_annotations.json
        """
        self.annotations = self._load_annotations(annotations_dir)
        self.X, self.y = self._load_all_npz(features_dir, self.annotations)

        # conversione a tensori
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).long()


    # -------------------------------------------------------------
    # 1) CARICAMENTO ANNOTAZIONI
    # -------------------------------------------------------------
    @staticmethod
    def _load_annotations(annotations_dir):
        """
        Carica esclusivamente il file 'complete_step_annotations.json'.
        """
        json_path = os.path.join(annotations_dir, "complete_step_annotations.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"File 'complete_step_annotations.json' non trovato in: {annotations_dir}"
            )

        with open(json_path, "r") as f:
            annotations = json.load(f)

        return annotations


    # -------------------------------------------------------------
    # 2) CARICAMENTO DI TUTTI GLI NPZ
    # -------------------------------------------------------------
    def _load_all_npz(self, features_dir, annotations):
        """
        Carica tutti i file .npz, genera le label e concatena tutto.
        """
        all_features = []
        all_labels = []

        for file in sorted(os.listdir(features_dir)):
            if not file.endswith(".npz"):
                continue
            
            file_path = os.path.join(features_dir, file)

            try:
                features, labels = self._get_labels_for_npz(file_path, annotations)
                all_features.append(features)
                all_labels.append(labels)
            except KeyError:
                # npz non presente nelle annotazioni
                print(f"[WARN] Nessuna annotazione trovata per: {file}")
                continue

        return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


    # -------------------------------------------------------------
    # 3) GENERAZIONE LABEL DA NPZ
    # -------------------------------------------------------------
    @staticmethod
    def _get_labels_for_npz(npz_file, annotations):
        """
        Genera le label per un singolo file .npz usando le annotazioni JSON.
        """

        # es: "10_3_360.mp4_1s_1s.npz" → recording_id = "10_3"
        base = os.path.basename(npz_file)
        activity, attempt = base.split("_")[:2]
        recording_id = f"{activity}_{attempt}"

        # features
        data = np.load(npz_file)
        arr = data[list(data.keys())[0]]  # shape (N, 1024)
        N = arr.shape[0]

        # default: tutti 0 = nessun errore
        labels = np.zeros(N, dtype=np.int64)

        # recupero annotazioni del video
        info = annotations[recording_id]
        steps = info["steps"]

        # assegnazione errore per ogni secondo
        for step in steps:
            has_error = int(step["has_errors"])
            start= step["start_time"]
            end= step["end_time"]

            # se non ci sono errori → skip
            if has_error == 0:
                continue

            # skip intervalli non validi
            if start == -1 or end == -1:
                continue

            for sec in range(int(start), int(end) + 1, 1):
                sec_start = sec
                sec_end   = sec + 1

                # check overlap
                if sec_start >= start and sec_end <= end: # i secondi ai bordi avranno sempre il valore di default (norml = no-error = 0)
                    labels[sec] = has_error
            
        return arr, labels


    # -------------------------------------------------------------
    # 4) METODI STANDARD DEL DATASET
    # -------------------------------------------------------------
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    # -------------------------------------------------------------
    # 5) METODO PER RESTITUIRE LO SHAPE
    # -------------------------------------------------------------
    def shape(self):
        """
        Restituisce una tupla (num_samples, num_features) delle feature X.
        """
        return self.X.shape