from enum import Enum
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from utils.setup_project import is_colab
from exceptions import (
    AnnotationNotFoundError,
    FeatureFileNotFoundError,
    EmptyDatasetError,
)

class DatasetSource(Enum):
    OMNIVORE = "omnivore"
    SLOWFAST = "slowfast"

class CaptainCook4D_Dataset(Dataset):
    """
    Dataset per CaptainCook4D basato su file .npz (es. feature Omnivore)
    e annotazioni JSON (complete_step_annotations.json).
    """
    
    def __init__(self, dataset_source: DatasetSource, root_dir: str):
        """
        Args:
            omnivore_dir (str): path alla cartella con i file .npz
            annotations_dir (str): path alla cartella con complete_step_annotations.json
        """
        self.dataset_source = dataset_source
        self.root_dir = root_dir

        print(f"Loading from: {self.features_dir()}...")

        self.annotations = self._load_annotations()
        self.X, self.y = self._load_all_npz(self.annotations)

        # conversione a tensori
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).long()
        

    def features_dir(self):
        # Combina la root con 'data' e il dataset_source
        return os.path.join(self.root_dir, "data", self.dataset_source.value)
    
    def annotations_dir(self):
        # Combina la root con 'data/annotation_json'
        return os.path.join(self.root_dir, "data", "annotation_json")

    # -------------------------------------------------------------
    # 1) CARICAMENTO ANNOTAZIONI
    # -------------------------------------------------------------
    def _load_annotations(self):
        """
        Carica esclusivamente il file 'complete_step_annotations.json'.
        """
        json_path = os.path.join(self.annotations_dir(), "complete_step_annotations.json")

        if not os.path.exists(json_path):
            raise AnnotationNotFoundError(
                annotation_file="complete_step_annotations.json",
                searched_path=self.annotations_dir()
            )

        try:
            with open(json_path, "r") as f:
                annotations = json.load(f)
        except json.JSONDecodeError as e:
            raise CorruptedFeatureFileError(
                file_path=json_path,
                original_error=f"JSON non valido: {e}"
            )

        return annotations


    # -------------------------------------------------------------
    # 2) CARICAMENTO DI TUTTI GLI NPZ
    # -------------------------------------------------------------
    def _load_all_npz(self, annotations):
        """
        Carica tutti i file .npz, genera le label e concatena tutto.
        """
        all_features = []
        all_labels = []

        for file in sorted(os.listdir(self.features_dir())):
            if not file.endswith(".npz"):
                continue
            
            file_path = os.path.join(self.features_dir(), file)

            try:
                features, labels = self._get_labels_for_npz(file_path, annotations)
                all_features.append(features)
                all_labels.append(labels)
            except KeyError as e:
                # npz non presente nelle annotazioni
                print(f"[WARN] Nessuna annotazione trovata per: {file}")
                continue
            except Exception as e:
                raise CorruptedFeatureFileError(
                    file_path=file_path,
                    original_error=str(e)
                )

        if not all_features:
            raise EmptyDatasetError(
                reason="Nessun file .npz valido trovato o caricato"
            )

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