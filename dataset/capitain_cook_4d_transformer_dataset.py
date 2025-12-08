import torch
from torch.utils.data import Dataset
import numpy as np
from .capitain_cook_4d_mlp_dataset import CaptainCook4DMLP_Dataset, DatasetSource


class CaptainCook4DTransformer_Dataset(Dataset):
    """
    Dataset per CaptainCook4D dove ogni record rappresenta uno step completo.
    
    A differenza del dataset base (dove ogni record è 1 secondo di uno step),
    qui ogni record contiene TUTTI i secondi di uno specifico step in un video.
    
    Ogni elemento X[i] ha shape: (durata_step_in_secondi, n_features)
    
    Nota: uno stesso step_id può apparire in video diversi, ma vengono 
    mantenuti come record separati (uno per ogni combinazione video_id + step_id).
    """
    
    def __init__(self, dataset_source: DatasetSource, root_dir: str):
        """
        Args:
            dataset_source (DatasetSource): fonte delle feature (OMNIVORE o SLOWFAST)
            root_dir (str): path alla cartella root del dataset
        """
        # Carica il dataset base (1 secondo per record)
        self.base_dataset = CaptainCook4DMLP_Dataset(dataset_source, root_dir)
        
        # Raggruppa i record per (video_id, step_id)
        self.X, self.y, self.step_ids, self.video_ids = self._group_by_steps()
        
        print(f"Dataset creato: {len(self)} step completi da {len(self.base_dataset)} secondi")
    
    def _group_by_steps(self):
        """
        Raggruppa i record del dataset base per (video_id, step_id).
        
        Returns:
            tuple: (X_list, y_list, step_ids_list, video_ids_list)
                - X_list: lista di matrici (durata_step, n_features)
                - y_list: lista di label (1 per step)
                - step_ids_list: lista di step_id
                - video_ids_list: lista di video_id
        """
        X_grouped = []
        y_grouped = []
        step_ids_grouped = []
        video_ids_grouped = []
        
        # Dizionario per raggruppare: chiave = (video_id, step_id)
        groups = {}
        
        # Raggruppa tutti i record per (video_id, step_id)
        for idx in range(len(self.base_dataset)):
            features, label, step_id, video_id = self.base_dataset[idx]
            
            key = (video_id, step_id)
            
            if key not in groups:
                groups[key] = {
                    'features': [],
                    'label': label.item()  # Assumiamo che tutti i secondi dello step abbiano la stessa label
                }
            
            groups[key]['features'].append(features.numpy())
        
        # Converte i gruppi in liste
        for (video_id, step_id), data in sorted(groups.items()):
            # Stack dei features: da lista di vettori a matrice (n_secondi, n_features)
            step_features = np.stack(data['features'], axis=0)
            
            X_grouped.append(torch.from_numpy(step_features).float())
            y_grouped.append(torch.tensor(data['label'], dtype=torch.long))
            step_ids_grouped.append(step_id)
            video_ids_grouped.append(video_id)
        
        return X_grouped, y_grouped, step_ids_grouped, video_ids_grouped
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Restituisce uno step completo.
        
        Returns:
            tuple: (features, label, step_id, video_id)
                - features: Tensor di shape (durata_step, n_features)
                - label: Tensor scalare (0=OK, 1=ERR)
                - step_id: int
                - video_id: str
        """
        return self.X[idx], self.y[idx], self.step_ids[idx], self.video_ids[idx]
    
    def shape(self):
        """
        Restituisce informazioni sulla forma del dataset.
        
        Returns:
            dict: informazioni sulla struttura
        """
        if len(self) == 0:
            return {
                'num_steps': 0,
                'n_features': 0,
                'min_duration': 0,
                'max_duration': 0,
                'avg_duration': 0.0
            }
        
        durations = [x.shape[0] for x in self.X]
        n_features = self.X[0].shape[1] if len(self.X) > 0 else 0
        
        return {
            'num_steps': len(self),
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
        X, y, step_id, video_id = self[idx]
        
        print("=" * 80)
        print(f"STEP DATASET ITEM [{idx}]")
        print("=" * 80)
        print(f"Features shape:       {X.shape} (durata_step, n_features)")
        print(f"Step duration:        {X.shape[0]} secondi")
        print(f"Label:                {y.item()} ({'OK' if y.item() == 0 else 'ERR'})")
        print(f"Step ID:              {step_id}")
        print(f"Video ID:             {video_id}")
        print("=" * 80)
    
    def get_step_info(self, video_id=None, step_id=None):
        """
        Restituisce gli indici dei record che corrispondono a un certo video_id e/o step_id.
        
        Args:
            video_id (str, optional): filtra per video_id
            step_id (int, optional): filtra per step_id
            
        Returns:
            list: lista di indici che corrispondono ai criteri
        """
        indices = []
        
        for idx in range(len(self)):
            match = True
            
            if video_id is not None and self.video_ids[idx] != video_id:
                match = False
            
            if step_id is not None and self.step_ids[idx] != step_id:
                match = False
            
            if match:
                indices.append(idx)
        
        return indices
    
    def print_summary(self):
        """
        Stampa un riassunto del dataset.
        """
        shape_info = self.shape()
        
        print("=" * 80)
        print("DATASET SUMMARY")
        print("=" * 80)
        print(f"Total steps:          {shape_info['num_steps']}")
        print(f"Features per second:  {shape_info['n_features']}")
        print(f"Step duration (min):  {shape_info['min_duration']} secondi")
        print(f"Step duration (max):  {shape_info['max_duration']} secondi")
        print(f"Step duration (avg):  {shape_info['avg_duration']:.2f} secondi")
        
        # Conta label
        num_errors = sum(1 for y in self.y if y.item() == 1)
        num_ok = sum(1 for y in self.y if y.item() == 0)
        
        print(f"\nLabel distribution:")
        print(f"  OK (0):             {num_ok} ({num_ok/len(self)*100:.1f}%)")
        print(f"  ERR (1):            {num_errors} ({num_errors/len(self)*100:.1f}%)")
        
        # Conta video unici
        unique_videos = len(set(self.video_ids))
        unique_steps = len(set(self.step_ids))
        
        print(f"\nUnique videos:        {unique_videos}")
        print(f"Unique step IDs:      {unique_steps}")
        print("=" * 80)
