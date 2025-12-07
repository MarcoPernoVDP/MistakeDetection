from enum import Enum
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from dataset.capitain_cook_4d_mlp_dataset import DatasetSource
from utils.setup_project import is_colab
from exceptions import (
    AnnotationNotFoundError,
    FeatureFileNotFoundError,
    EmptyDatasetError,
    CorruptedFeatureFileError,
)

class CaptainCook4DTransformer_Dataset(Dataset):
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
        
        self.X, self.y, self.step_ids = self._load_all_npz(self.annotations)
        # conversione a tensori
        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).long()
        self.step_ids = self.step_ids  # Mantieni come numpy array di stringhe
            
        # V2: crea lista di step_ids univoci FILTRANDO I NONE
        valid_step_ids = self.step_ids[self.step_ids != None]
        self.unique_step_ids = np.unique(valid_step_ids).tolist()
        
        print(f"[V2] Numero di step univoci: {len(self.unique_step_ids)}")

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
        all_step_ids = []

        for file in sorted(os.listdir(self.features_dir())):
            if not file.endswith(".npz"):
                continue
            
            file_path = os.path.join(self.features_dir(), file)

            try:
                result = self._get_labels_for_npz(file_path, annotations)
                
                features, labels, step_ids = result
                all_step_ids.append(step_ids)
                
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

        return (np.concatenate(all_features, axis=0), 
                np.concatenate(all_labels, axis=0),
                np.concatenate(all_step_ids, axis=0))


    # -------------------------------------------------------------
    # 3) GENERAZIONE LABEL DA NPZ
    # -------------------------------------------------------------
    def _get_labels_for_npz(self, npz_file, annotations):
        """
        Genera le label (e step_ids se V2) per un singolo file .npz usando le annotazioni JSON.
        
        Returns:
            V1: tuple (features, labels)
            V2: tuple (features, labels, step_ids)
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
        step_ids = np.empty(N, dtype=object)

        # recupero annotazioni del video
        info = annotations[recording_id]
        steps = info["steps"]

        # assegnazione errore (e step_id per V2) per ogni secondo
        for step_idx, step in enumerate(steps):
            has_error = int(step["has_errors"])
            start = step.get("start_time", -1)
            end = step.get("end_time", -1)

            # skip intervalli non validi
            if start == -1 or end == -1:
                continue

            # V2: step_id univoco che include timing
            # formato: "{recording_id}_{step_idx}_{start_time:.3f}_{end_time:.3f}"
            # es: "10_3_5_7.072_46.288"
            step_id = f"{recording_id}_{step_idx}_{start:.3f}_{end:.3f}"

            # Assegna label e step_id per TUTTI i secondi di questo step
            for sec in range(int(start), int(end) + 1, 1):
                if sec < N:  # check boundary
                    labels[sec] = has_error
                    
                    # V2: traccia lo step di appartenenza per TUTTI gli step
                    step_ids[sec] = step_id

        return arr, labels, step_ids


    # -------------------------------------------------------------
    # 4) METODI STANDARD DEL DATASET
    # -------------------------------------------------------------
    def __len__(self):
        return len(self.unique_step_ids)

    def __getitem__(self, idx):
        # V2: restituisce TUTTI i sotto-step di uno step
        step_id = self.unique_step_ids[idx]
        
        # Converti esplicitamente a numpy array di stringhe per la comparazione
        step_ids_array = np.array(self.step_ids, dtype=str)
        mask = step_ids_array == step_id
        indices = np.where(mask)[0]
        
        # Ritorna: (X_seq, y_seq, step_id, num_substeps)
        return (
            self.X[indices],           # sequenza di feature
            self.y[indices],           # sequenza di label
            step_id,                   # ID dello step (stringa)
            len(indices)               # lunghezza della sequenza
        )
    
    # -------------------------------------------------------------
    # 5) METODI SPECIFICI PER V2
    # -------------------------------------------------------------
    def get_substeps_for_step(self, step_id):
        """
        [Solo V2] Restituisce tutti i sotto-step da 1s che appartengono a uno specifico step.
        
        Args:
            step_id: ID dello step (formato: recording_id_numeric * 1000 + step_index)
            
        Returns:
            tuple: (X, y, step_ids, indices) - features, labels, step_ids e indici dei sotto-step
        """
        mask = self.step_ids == step_id
        indices = torch.where(mask)[0]
        return self.X[indices], self.y[indices], self.step_ids[indices], indices.numpy()
    
    def get_steps_for_recording(self, recording_id):
        """
        [Solo V2] Restituisce tutti gli step_ids univoci per una specifica registrazione.
        
        Args:
            recording_id: ID della registrazione come stringa (es: "10_3")
            
        Returns:
            list: lista degli step_ids univoci (stringhe come "10_3_0", "10_3_1", ...)
        """
        mask = np.array([sid.startswith(recording_id) for sid in self.step_ids])
        unique_steps = np.unique(self.step_ids[mask])
        return unique_steps.tolist()

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
        X_seq, y_seq, step_id, seq_len = item
        
        print("=" * 80)
        print(f"V2 DATASET ITEM [{idx}]")
        print("=" * 80)
        print(f"Step ID:              {step_id}")
        print(f"Sequence length:      {seq_len} seconds")
        print(f"Features shape:       {X_seq.shape} (seconds x features)")
        print(f"Labels shape:         {y_seq.shape} (seconds)")
        
        # Parse step_id
        parts = step_id.split("_")
        recording_id = f"{parts[0]}_{parts[1]}"
        step_idx = parts[2]
        start_time = float(parts[3])
        end_time = float(parts[4])
        
        print(f"\nStep details:")
        print(f"  Video ID:             {recording_id}")
        print(f"  Step index:           {step_idx}")
        print(f"  Timing:               {start_time:.3f}s - {end_time:.3f}s ({end_time - start_time:.3f}s)")
        print(f"  Label sequence:       {y_seq.tolist()}")
        print("=" * 80)