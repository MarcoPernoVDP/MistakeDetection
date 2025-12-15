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
    CorruptedFeatureFileError,
)

class DatasetSource(Enum):
    OMNIVORE = "omnivore"
    SLOWFAST = "slowfast"

    def input_dims(self) -> int:
        if self == DatasetSource.OMNIVORE:
            return 1024
        elif self == DatasetSource.SLOWFAST:
            return 400

class CaptainCook4DMLP_Dataset(Dataset):
    """
    Dataset per CaptainCook4D basato su file .npz (es. feature Omnivore)
    e annotazioni JSON (complete_step_annotations.json).
    """
    
    def __init__(self, dataset_source: DatasetSource, root_dir: str):
        """
        Args:
            dataset_source (DatasetSource): fonte delle feature (OMNIVORE o SLOWFAST)
            root_dir (str): path alla cartella root del dataset
        """
        self.dataset_source = dataset_source
        self.root_dir = root_dir

        print(f"Loading from: {self.features_dir()}...")

        self.annotations = self._load_annotations()
        
        self.X, self.y, self.steps, self.videos, self.start_times = self._load_all_npz(self.annotations)
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
        Carica tutti i file .npz, genera le label (e step_ids se V2) e concatena tutto.
        """
        all_features = []
        all_labels = []
        all_steps = []
        all_videos = []
        all_start_times = []

        for file in sorted(os.listdir(self.features_dir())):
            if not file.endswith(".npz"):
                continue
            
            file_path = os.path.join(self.features_dir(), file)

            try:
                result = self._get_labels_for_npz(file_path, annotations)
                
                features, labels, steps, videos, start_times = result
                
                all_features.append(features)
                all_labels.append(labels)
                all_steps.append(steps)
                all_videos.append(videos)
                all_start_times.append(start_times)
                
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

        return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0), np.concatenate(all_steps, axis=0), np.concatenate(all_videos, axis=0), np.concatenate(all_start_times, axis=0)


    # -------------------------------------------------------------
    # 3) GENERAZIONE LABEL DA NPZ
    # -------------------------------------------------------------
    def _get_labels_for_npz(self, npz_file, annotations):
        """
        Genera le label (e step_ids se V2) per un singolo file .npz usando le annotazioni JSON.
        
        Returns:
            tuple (features, labels, steps, videos, start_times)
        """

        # es: "10_3_360.mp4_1s_1s.npz" → recording_id = "10_3"
        base = os.path.basename(npz_file)
        activity, attempt = base.split("_")[:2]
        recording_id = f"{activity}_{attempt}"

        # features
        data = np.load(npz_file)
        arr = data[list(data.keys())[0]]  # shape (N, 1024)
        N = arr.shape[0]

        # default: tutti -1 -> non classificati
        labels = np.ones(N, dtype=np.int64) * -1
        steps = np.ones(N, dtype=np.int64) * -1
        videos = np.empty(N, dtype=object)
        start_times = np.empty(N, dtype=np.float64)

        # recupero annotazioni del video
        info = annotations[recording_id]
        steps_dict = info["steps"]

        # assegnazione errore (e step_id per V2) per ogni secondo
        for step_idx, step_dict in enumerate(steps_dict):
            has_error = int(step_dict["has_errors"])
            start = step_dict.get("start_time", -1)
            end = step_dict.get("end_time", -1)

            # skip intervalli non validi
            if start == -1 or end == -1:
                continue

            # Assegna label e step_id per TUTTI i secondi di questo step
            for sec in range(int(start), int(end) + 1, 1):
                if sec < N:  # check boundary
                    labels[sec] = has_error
                    steps[sec] = step_dict["step_id"]
                    videos[sec] = recording_id
                    start_times[sec] = start


        # Rimuovi i records che non fanno parte di uno step
        # Crea una maschera per tenere solo i records che hanno una label valida (0 o 1)
        valid_mask = (labels == 0) | (labels == 1)
        arr = arr[valid_mask]
        labels = labels[valid_mask]
        steps = steps[valid_mask]
        videos = videos[valid_mask]
        start_times = start_times[valid_mask]

        return arr, labels, steps, videos, start_times


    # -------------------------------------------------------------
    # 4) METODI STANDARD DEL DATASET
    # -------------------------------------------------------------
    def __len__(self):
        # V1: numero di sotto-step da 1s
        return len(self.X)

    def __getitem__(self, idx):
        # V1: restituisce singolo sotto-step
        return self.X[idx], self.y[idx], self.steps[idx], self.videos[idx], self.start_times[idx]
    
    # -------------------------------------------------------------
    # 6) METODO PER RESTITUIRE LO SHAPE
    # -------------------------------------------------------------
    def shape(self):
        """
        Restituisce una tupla (num_samples, num_features) delle feature X.
        """
        return self.X.shape

    # -------------------------------------------------------------
    # 7) METODI PER LA STAMPA
    # -------------------------------------------------------------
    def print_item(self, idx):
        """
        Stampa formattata di un elemento del dataset.
        
        Args:
            idx: indice dell'elemento
        """
        item = self[idx]
        
        X, y, step_id, video_id = item
            
        print("=" * 80)
        print(f"V1 DATASET ITEM [{idx}]")
        print("=" * 80)
        print(f"Features shape:       {X.shape} (features)")
        print(f"Label:                {y.item()} ({'OK' if y.item() == 0 else 'ERR'})")
        print(f"Step id:              {step_id}")
        print(f"Video id:             {video_id}")
        print(f"Start time:           {self.start_times[idx]} seconds")
        print("=" * 80)