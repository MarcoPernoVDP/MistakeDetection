# feature_extractor.py
import os
import glob
import numpy as np
import torch
import timm
from decord import VideoReader, cpu
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import Dataset, DataLoader

@dataclass
class ExtractionConfig:
    """Configurazione per l'estrazione delle feature."""
    video_dir: str              
    output_root: str            
    model_name: str = 'vit_base_patch16_224.fb_pe_core' 
    batch_size: int = 64        # Batch per l'inferenza GPU
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    fps_target: int = 1         
    num_workers: int = 4        # NUOVO: Numero di processi paralleli per caricare i video

class VideoLoader:
    """Gestisce il preprocessing dei frame."""
    def __init__(self, model_config):
        self.transform = create_transform(**model_config)
    
    def process_video(self, video_path: str) -> Optional[torch.Tensor]:
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            video_fps = vr.get_avg_fps()
            duration = len(vr) / video_fps
            frames_tensor_list = []
            
            num_seconds = int(np.ceil(duration))
            for sec in range(num_seconds):
                frame_idx = int(min((sec + 0.5) * video_fps, len(vr) - 1))
                frame_np = vr[frame_idx].asnumpy()
                frame_pil = Image.fromarray(frame_np)
                frames_tensor_list.append(self.transform(frame_pil))
            
            if not frames_tensor_list:
                return None
                
            return torch.stack(frames_tensor_list)
        except Exception as e:
            # Stampiamo l'errore ma non blocchiamo il worker
            print(f"⚠️ Errore lettura {os.path.basename(video_path)}: {e}")
            return None

class VideoDataset(Dataset):
    """Dataset PyTorch per caricare i video in parallelo."""
    def __init__(self, video_paths, loader, save_dir):
        self.video_paths = video_paths
        self.loader = loader
        self.save_dir = save_dir

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        filename = os.path.basename(video_path)
        save_path = os.path.join(self.save_dir, f"{filename}_1s_1s.npz")
        
        # Check esistenza file (Resume) fatto nel worker
        if os.path.exists(save_path):
            return None # Skip segnalato con None
            
        video_tensor = self.loader.process_video(video_path)
        
        if video_tensor is None:
            return None # Skip per errore lettura
            
        return video_tensor, save_path

def collate_fn_skip_none(batch):
    """
    Gestisce il caso in cui un video sia skippato (None).
    Poiché usiamo batch_size=1 nel DataLoader, 'batch' è una lista di 1 elemento.
    """
    item = batch[0]
    if item is None:
        return None
    return item

class PerceptionFeatureExtractor:
    """Motore principale per l'estrazione delle feature."""
    
    def __init__(self, config: ExtractionConfig):
        self.cfg = config
        self._setup_model()
        # Inizializza il loader che verrà passato ai worker
        self.loader = VideoLoader(resolve_data_config({}, model=self.model))
        
        self.save_dir = os.path.join(
            self.cfg.output_root, "perception_encoder", "segment", "1s"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Directory Output configurata: {self.save_dir}")

    def _setup_model(self):
        print(f"Caricamento Modello: {self.cfg.model_name}...")
        self.model = timm.create_model(
            self.cfg.model_name, pretrained=True, num_classes=0
        )
        self.model.to(self.cfg.device).eval()
        print("✅ Modello pronto.")

    def run(self):
        # 1. Trova i file video
        search_pattern = os.path.join(self.cfg.video_dir, "**", "*.mp4")
        videos = glob.glob(search_pattern, recursive=True)
        
        if not videos:
            print(f"❌ Nessun video trovato in {self.cfg.video_dir}")
            return

        # 2. Crea Dataset e DataLoader
        # Passiamo il loader esistente al dataset
        dataset = VideoDataset(videos, self.loader, self.save_dir)
        
        # batch_size=1 nel DataLoader perché i video hanno lunghezza diversa.
        # Il parallelismo reale avviene grazie a num_workers che pre-carica i prossimi N video.
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn_skip_none,
            pin_memory=True if self.cfg.device == 'cuda' else False
        )

        print(f"Avvio estrazione con {self.cfg.num_workers} workers...")

        # 3. Ciclo di inferenza
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(videos), desc="Estrazione Feature"):
                # Se il worker ha ritornato None (video già esistente o errore), saltiamo
                if batch is None:
                    continue
                
                video_tensor, save_path = batch
                
                # video_tensor arriva dal loader con dimensione [T, C, H, W]
                # Non c'è la dimensione batch extra perché collate_fn l'ha rimossa o non aggiunta
                
                # Inferenza a blocchi (GPU Batching)
                features_list = []
                for i in range(0, len(video_tensor), self.cfg.batch_size):
                    batch_gpu = video_tensor[i : i + self.cfg.batch_size].to(self.cfg.device)
                    emb = self.model(batch_gpu)
                    features_list.append(emb.cpu().numpy())

                # Salvataggio
                if features_list:
                    final_features = np.vstack(features_list)
                    np.savez_compressed(save_path, features=final_features)
                
        print("Estrazione Completata!")