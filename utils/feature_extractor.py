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

@dataclass
class ExtractionConfig:
    """Configurazione per l'estrazione delle feature."""
    video_dir: str              # Cartella dove si trovano i video .mp4
    output_root: str            # Cartella radice dove salvare le feature
    model_name: str = 'vit_base_patch16_224.fb_pe_core' # Modello Perception Encoder
    batch_size: int = 64        # Dimensione batch per la GPU
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    fps_target: int = 1         # Quante feature per secondo estrarre (Default: 1)
    
class VideoLoader:
    """Gestisce il caricamento video e il preprocessing dei frame."""
    
    def __init__(self, model_config):
        # Crea le trasformazioni (resize, norm) specifiche del modello
        self.transform = create_transform(**model_config)
    
    def process_video(self, video_path: str) -> Optional[torch.Tensor]:
        try:
            # Decord: lettura efficiente senza decoding completo
            vr = VideoReader(video_path, ctx=cpu(0))
            video_fps = vr.get_avg_fps()
            duration = len(vr) / video_fps
            frames_tensor_list = []
            
            # Logica: 1 frame al centro di ogni secondo
            num_seconds = int(np.ceil(duration))
            for sec in range(num_seconds):
                # Calcola indice frame
                frame_idx = int(min((sec + 0.5) * video_fps, len(vr) - 1))
                
                # Converti in PIL e applica trasformazioni
                frame_np = vr[frame_idx].asnumpy()
                frame_pil = Image.fromarray(frame_np)
                frames_tensor_list.append(self.transform(frame_pil))
            
            if not frames_tensor_list:
                return None
                
            # Restituisce tensore (T, C, H, W)
            return torch.stack(frames_tensor_list)
            
        except Exception as e:
            print(f"⚠️ Errore lettura {os.path.basename(video_path)}: {e}")
            return None

class PerceptionFeatureExtractor:
    """Motore principale per l'estrazione delle feature con Perception Encoder."""
    
    def __init__(self, config: ExtractionConfig):
        self.cfg = config
        self._setup_model()
        self.loader = VideoLoader(resolve_data_config({}, model=self.model))
        
        # Struttura output standard per il progetto:
        # {output_root}/perception_encoder/segment/1s/
        self.save_dir = os.path.join(
            self.cfg.output_root, 
            "perception_encoder", 
            "segment", 
            "1s"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Directory Output configurata: {self.save_dir}")

    def _setup_model(self):
        print(f"Caricamento Modello: {self.cfg.model_name}...")
        # num_classes=0 rimuove l'ultimo layer per ottenere l'embedding puro
        self.model = timm.create_model(
            self.cfg.model_name, 
            pretrained=True, 
            num_classes=0
        )
        self.model.to(self.cfg.device).eval()
        print("✅ Modello pronto.")

    def run(self):
        """Esegue l'estrazione su tutti i video nella cartella configurata."""
        # Cerca video ricorsivamente (.mp4)
        search_pattern = os.path.join(self.cfg.video_dir, "**", "*.mp4")
        videos = glob.glob(search_pattern, recursive=True)
        
        if not videos:
            print(f"❌ Nessun video trovato in {self.cfg.video_dir}")
            return
        
        for video_path in tqdm(videos, desc="Estrazione Feature"):
            filename = os.path.basename(video_path)
            # Nome file output: video.mp4_1s_1s.npz
            save_path = os.path.join(self.save_dir, f"{filename}_1s_1s.npz")
            
            # Skip se esiste già (Resume automatico)
            if os.path.exists(save_path): 
                continue

            # 1. Carica e trasforma i frame
            video_tensor = self.loader.process_video(video_path)
            if video_tensor is None: 
                continue

            # 2. Inferenza in Batch (per non saturare la VRAM)
            features_list = []
            with torch.no_grad():
                for i in range(0, len(video_tensor), self.cfg.batch_size):
                    batch = video_tensor[i : i + self.cfg.batch_size].to(self.cfg.device)
                    # Forward pass
                    emb = self.model(batch)
                    features_list.append(emb.cpu().numpy())

            # 3. Salvataggio
            if features_list:
                final_features = np.vstack(features_list)
                np.savez_compressed(save_path, features=final_features)
                
        print("Estrazione Completata!")