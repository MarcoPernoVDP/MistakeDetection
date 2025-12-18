# feature_extractor.py
import os
import gc
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
        dataset = VideoDataset(videos, self.loader, self.save_dir)
        
        # MODIFICA CRITICA 1: Riduciamo la pressione sul DataLoader
        # Su Colab con poca RAM, pin_memory può essere dannoso se la RAM è al limite.
        # num_workers > 0 copia i dati in memoria condivisa, che consuma molta RAM.
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.cfg.num_workers, # Consiglio: metti a 0 o 1 nel Config se crasha ancora
            collate_fn=collate_fn_skip_none,
            pin_memory=False # Mettiamo False per sicurezza su Colab
        )

        print(f"Avvio estrazione con {self.cfg.num_workers} workers...")

        # 3. Ciclo di inferenza
        with torch.no_grad():
            # Usiamo enumerate per poter chiamare il GC ogni tot cicli
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(videos), desc="Estrazione Feature"):
                
                if batch is None:
                    continue
                
                video_tensor, save_path = batch
                
                # MODIFICA CRITICA 2: Gestione Tensori
                # video_tensor qui è [1, T, C, H, W] (per via del batch_size=1 del loader)
                # Lo rimuoviamo dalla dimensione batch inutile per lavorarci meglio
                if video_tensor.dim() == 5:
                    video_tensor = video_tensor.squeeze(0)
                
                features_list = []
                
                try:
                    # Inferenza a blocchi (GPU Batching)
                    for i in range(0, len(video_tensor), self.cfg.batch_size):
                        # Spostiamo su GPU SOLO il pezzettino che serve ora
                        batch_gpu = video_tensor[i : i + self.cfg.batch_size].to(self.cfg.device, non_blocking=True)
                        
                        emb = self.model(batch_gpu)
                        
                        # Spostiamo subito su CPU e in Numpy per liberare VRAM
                        features_list.append(emb.cpu().numpy())
                        
                        # Pulizia variabili temporanee loop
                        del batch_gpu, emb

                    # Salvataggio
                    if features_list:
                        final_features = np.vstack(features_list)
                        np.savez_compressed(save_path, features=final_features)
                
                except Exception as e:
                    print(f"Errore durante inferenza: {e}")
                
                finally:
                    # MODIFICA CRITICA 3: Pulizia Aggressiva della Memoria
                    # Cancelliamo esplicitamente le variabili grandi
                    del video_tensor
                    del features_list
                    if 'final_features' in locals(): del final_features
                    if 'batch' in locals(): del batch
                    
                    # Svuota la cache CUDA
                    torch.cuda.empty_cache()
                    
                    # Forza il Garbage Collector di Python ogni 5 video (o anche 1 se necessario)
                    if batch_idx % 5 == 0:
                        gc.collect()

        print("Estrazione Completata!")