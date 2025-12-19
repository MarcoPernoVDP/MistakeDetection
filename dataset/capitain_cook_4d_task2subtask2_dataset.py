import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from glob import glob


class CaptainCook4DTask2Subtask2_Dataset(Dataset):
    """
    Dataset per CaptainCook4D dove ogni record rappresenta un video completo.
    
    Ogni record contiene TUTTI gli step di un intero video.
    Carica i dati direttamente dalla cartella data/hiero dove ogni file .npz
    contiene gli step di un video.
    
    Ogni elemento X[i] ha shape: (num_steps, n_features)
    
    La label indica se il video contiene errori (1) o no (0), basandosi
    sulle annotazioni a livello video.
    """
    
    def __init__(self, root_dir: str):
        """
        Args:
            root_dir (str): path alla cartella root del dataset
        """
        self.root_dir = root_dir
        
        print(f"Loading from: {self.features_dir()}...")
        
        # Carica le annotazioni a livello video
        self.video_annotations = self._load_video_annotations(root_dir)
        
        # Carica i dati da hiero
        self.X, self.y, self.video_ids = self._load_from_hiero()
        
        print(f"Dataset creato: {len(self)} video completi")
    
    def features_dir(self) -> str:
        return os.path.join(self.root_dir, 'data', 'hiero')
    
    def annotations_dir(self) -> str:
        return os.path.join(self.root_dir, 'data', 'annotation_json')
    
    def _load_video_annotations(self, root_dir: str):
        """
        Carica le annotazioni a livello video dal file JSON.
        
        Args:
            root_dir (str): path alla cartella root del dataset
            
        Returns:
            dict: dizionario con video_id come chiave e has_errors come valore
        """
        json_path = os.path.join(root_dir, 'data', 'annotation_json', 'video_level_annotations.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Estrai solo il campo has_errors per ogni video
        video_annotations = {video_id: info['has_errors'] for video_id, info in data.items()}
        return video_annotations

    def _load_from_hiero(self):
        """
        Carica i dati dai file .npz nella cartella hiero.
        Ogni file corrisponde a un video e contiene tutti i suoi step.
        
        Returns:
            tuple: (X_list, y_list, video_ids_list)
                - X_list: lista di matrici (num_steps, n_features)
                - y_list: lista di label (0=no errors, 1=has errors)
                - video_ids_list: lista di video_id
        """
        X_list = []
        y_list = []
        video_ids_list = []
        
        # Trova tutti i file .npz nella cartella hiero
        hiero_dir = self.features_dir()
        npz_files = sorted(glob(os.path.join(hiero_dir, '*.npz')))
        
        if len(npz_files) == 0:
            print(f"Warning: No .npz files found in {hiero_dir}")
            return [], [], []
        
        for npz_path in npz_files:
            # Estrai il video_id dal nome del file
            # Esempio: "1_10_360p.mp4_1s_1s_steps.npz" -> "1_10_360"
            filename = os.path.basename(npz_path)
            video_id = filename.replace('_360p.mp4_1s_1s_steps.npz', '')
            
            # Carica il file npz
            data = np.load(npz_path, allow_pickle=True)['data']
            
            # Estrai gli embeddings da tutti gli step
            step_embeddings = []
            for step_data in data:
                embedding = step_data['embedding']
                step_embeddings.append(embedding)
            
            if len(step_embeddings) == 0:
                print(f"Warning: No steps found in {filename}, skipping...")
                continue
            
            # Stack degli embeddings: (num_steps, n_features)
            video_features = np.stack(step_embeddings, axis=0)
            
            # Determina la label dal file JSON
            has_errors = self.video_annotations.get(video_id, False)
            label = 1 if has_errors else 0
            
            X_list.append(torch.from_numpy(video_features).float())
            y_list.append(torch.tensor(label, dtype=torch.long))
            video_ids_list.append(video_id)
        
        return X_list, y_list, video_ids_list
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Restituisce un video completo.
        
        Returns:
            tuple: (features, label, video_id)
                - features: Tensor di shape (durata_video, n_features)
                - label: Tensor scalare (0=no errors, 1=has errors)
                - video_id: str
        """
        return self.X[idx], self.y[idx], self.video_ids[idx]
    
    def shape(self):
        """
        Restituisce informazioni sulla forma del dataset.
        
        Returns:
            dict: informazioni sulla struttura
        """
        if len(self) == 0:
            return {
                'num_videos': 0,
                'n_features': 0,
                'min_steps': 0,
                'max_steps': 0,
                'avg_steps': 0.0
            }
        
        num_steps = [x.shape[0] for x in self.X]
        n_features = self.X[0].shape[1] if len(self.X) > 0 else 0
        
        return {
            'num_videos': len(self),
            'n_features': n_features,
            'min_steps': min(num_steps),
            'max_steps': max(num_steps),
            'avg_steps': np.mean(num_steps)
        }
    
    def print_item(self, idx):
        """
        Stampa formattata di un elemento del dataset.
        
        Args:
            idx: indice dell'elemento
        """
        X, y, video_id = self[idx]
        
        print("=" * 80)
        print(f"VIDEO DATASET ITEM [{idx}]")
        print("=" * 80)
        print(f"Features shape:       {X.shape} (num_steps, n_features)")
        print(f"Number of steps:      {X.shape[0]}")
        print(f"Label:                {y.item()} ({'No Errors' if y.item() == 0 else 'Has Errors'})")
        print(f"Video ID:             {video_id}")
        print("=" * 80)
    
    def print_summary(self):
        """
        Stampa un riassunto del dataset.
        """
        shape_info = self.shape()
        
        print("=" * 80)
        print("DATASET SUMMARY")
        print("=" * 80)
        print(f"Total videos:         {shape_info['num_videos']}")
        print(f"Features per step:    {shape_info['n_features']}")
        print(f"Steps per video (min):{shape_info['min_steps']}")
        print(f"Steps per video (max):{shape_info['max_steps']}")
        print(f"Steps per video (avg):{shape_info['avg_steps']:.2f}")
        
        # Conta label
        num_errors = sum(1 for y in self.y if y.item() == 1)
        num_ok = sum(1 for y in self.y if y.item() == 0)
        
        print(f"\nLabel distribution:")
        print(f"  No Errors (0):      {num_ok} ({num_ok/len(self)*100:.1f}%)")
        print(f"  Has Errors (1):     {num_errors} ({num_errors/len(self)*100:.1f}%)")
        
        print("=" * 80)
