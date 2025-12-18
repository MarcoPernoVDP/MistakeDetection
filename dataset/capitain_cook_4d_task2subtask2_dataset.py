import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from .capitain_cook_4d_mlp_dataset import CaptainCook4DMLP_Dataset, DatasetSource


class CaptainCook4DTask2Subtask2_Dataset(Dataset):
    """
    Dataset per CaptainCook4D dove ogni record rappresenta un video completo.
    
    A differenza del dataset base (dove ogni record è 1 secondo di uno step),
    qui ogni record contiene TUTTI i secondi di un intero video.
    
    Ogni elemento X[i] ha shape: (durata_video_in_secondi, n_features)
    
    La label indica se il video contiene errori (1) o no (0), basandosi
    sulle annotazioni a livello video.
    """
    
    def __init__(self, dataset_source: DatasetSource, root_dir: str):
        """
        Args:
            dataset_source (DatasetSource): fonte delle feature (OMNIVORE o SLOWFAST)
            root_dir (str): path alla cartella root del dataset
        """
        # Carica il dataset base (1 secondo per record)
        self.base_dataset = CaptainCook4DMLP_Dataset(dataset_source, root_dir)
        
        # Carica le annotazioni a livello video
        self.video_annotations = self._load_video_annotations(root_dir)
        
        # Raggruppa i record per video_id
        self.X, self.y, self.video_ids = self._group_by_videos()
        
        print(f"Dataset creato: {len(self)} video completi da {len(self.base_dataset)} secondi")
    
    def features_dir(self) -> str:
        return self.base_dataset.features_dir()
    
    def annotations_dir(self) -> str:
        return self.base_dataset.annotations_dir()
    
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

    def _group_by_videos(self):
        """
        Raggruppa i record del dataset base per video_id.
        
        Returns:
            tuple: (X_list, y_list, video_ids_list)
                - X_list: lista di matrici (durata_video, n_features)
                - y_list: lista di label (0=no errors, 1=has errors)
                - video_ids_list: lista di video_id
        """
        X_grouped = []
        y_grouped = []
        video_ids_grouped = []
        
        # Dizionario per raggruppare: chiave = video_id
        groups = {}
        
        # Raggruppa tutti i record per video_id
        for idx in range(len(self.base_dataset)):
            features, label, step_id, video_id, start_time = self.base_dataset[idx]
            
            if video_id not in groups:
                groups[video_id] = {'features': []}
            
            groups[video_id]['features'].append(features.numpy())
        
        # Converte i gruppi in liste
        for video_id in sorted(groups.keys()):
            # Stack dei features: da lista di vettori a matrice (n_secondi, n_features)
            video_features = np.stack(groups[video_id]['features'], axis=0)
            
            # Determina la label dal file JSON
            has_errors = self.video_annotations.get(video_id, False)
            label = 1 if has_errors else 0
            
            X_grouped.append(torch.from_numpy(video_features).float())
            y_grouped.append(torch.tensor(label, dtype=torch.long))
            video_ids_grouped.append(video_id)
        
        return X_grouped, y_grouped, video_ids_grouped
    
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
                'min_duration': 0,
                'max_duration': 0,
                'avg_duration': 0.0
            }
        
        durations = [x.shape[0] for x in self.X]
        n_features = self.X[0].shape[1] if len(self.X) > 0 else 0
        
        return {
            'num_videos': len(self),
            'n_features': n_features,
            'min_duration': min(durations),
            'max_duration': max(durations),
            'avg_duration': np.mean(durations)
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
        print(f"Features shape:       {X.shape} (durata_video, n_features)")
        print(f"Video duration:       {X.shape[0]} secondi")
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
        print(f"Features per second:  {shape_info['n_features']}")
        print(f"Video duration (min): {shape_info['min_duration']} secondi")
        print(f"Video duration (max): {shape_info['max_duration']} secondi")
        print(f"Video duration (avg): {shape_info['avg_duration']:.2f} secondi")
        
        # Conta label
        num_errors = sum(1 for y in self.y if y.item() == 1)
        num_ok = sum(1 for y in self.y if y.item() == 0)
        
        print(f"\nLabel distribution:")
        print(f"  No Errors (0):      {num_ok} ({num_ok/len(self)*100:.1f}%)")
        print(f"  Has Errors (1):     {num_errors} ({num_errors/len(self)*100:.1f}%)")
        
        print("=" * 80)
