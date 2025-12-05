import os
import sys
import subprocess
import torch

def is_colab():
    return 'google.colab' in sys.modules

def get_colab_secret(key):
    if is_colab():
        try:
            from google.colab import userdata
            return userdata.get(key)
        except: return None
    return os.environ.get(key)

# def install_deps(root_dir):
#     """Installa dipendenze (solo su Colab)"""
#     req_file = os.path.join(root_dir, 'requirements.txt')
#     if os.path.exists(req_file):
#         print("📦 Installazione PyTorch con CUDA 12.4...")
#         subprocess.check_call([
#             sys.executable, "-m", "pip", "install",
#             "torch==2.9.0+cu124",
#             "torchvision==0.24.0+cu124",
#             "torchaudio==2.9.0",
#             "--index-url", "https://download.pytorch.org/whl/cu124"
#         ])

#         print("📦 Installazione delle altre librerie dal requirements...")
#         subprocess.check_call([
#             sys.executable, "-m", "pip", "install", "-r", req_file
#         ])

#         print("✅ Installazione completata!")
#     else:
#         print("requirements.txt non trovato")

def setup_wandb():
    """Login WandB"""
    key = get_colab_secret('WANDB_API_KEY')
    if key:
        import wandb
        wandb.login(key=key)
        print("WandB Logged in.")
    else:
        print("ℹWandB: Chiave non trovata (Offline/Manual).")

def initialize(root_dir):
    """
    Funzione Unica di Inizializzazione Progetto.
    Gestisce il caso di cartella dati vuota o inesistente.
    """
    print(f"Setup Progetto in: {root_dir}")
    
    setup_skipped = False
    
    if is_colab():
        source_path = "/content/drive/MyDrive/MistakeDetection"
    else:
        source_path = os.path.join(root_dir, "_file")

    print(f"source_path: {source_path}")
    
    if os.path.exists(source_path) and len(os.listdir(source_path)) > 0:
        try:
            from .setup_dataset import setup_dataset
            print(f"Setup Dati da: {source_path}")
            setup_dataset(source_path, root_dir)
        except FileNotFoundError as e:
            print(f"Errore durante il setup dati: {e}")
            print("Controlla che i file zip o le cartelle siano dentro '_file'.")
            setup_skipped = True
        except Exception as e:
            print(f"Errore generico setup dati: {e}")
            setup_skipped = True
    else:
        print(f"Sorgente dati '{source_path}' non trovata o vuota.")
        print("Se sei in locale, usare la cartella '_file'")
        setup_skipped = True

    # 3. WandB
    # setup_wandb()
    
    # 4. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if setup_skipped:
        print("\n⚠️ ATTENZIONE: Il setup dei dati è stato saltato.")
        print("Il codice funzionerà, ma il Dataloader fallirà se provi ad addestrare senza dati.")
    
    return device