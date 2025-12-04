import os
import shutil
import zipfile
from tqdm import tqdm  # Opzionale, per barra di progresso se installata

def setup_dataset(source_path, project_root, data_dir_name="data"):
    """
    Copia cartelle ed estrae zip dalla 'source_path' alla cartella 'data/' del progetto.
    Robusto: gestisce cartelle mancanti e sovrascritture.
    """
    # 1. Definisci e crea la cartella di destinazione
    dest_data_dir = os.path.join(project_root, data_dir_name)
    os.makedirs(dest_data_dir, exist_ok=True)

    print(f"Inizio setup dati...")
    print(f"   Sorgente: {source_path}")
    print(f"   Destinazione: {dest_data_dir}")

    # 2. Controllo Preliminare
    if not os.path.exists(source_path):
        print(f"⚠️ ATTENZIONE: La cartella sorgente '{source_path}' non esiste.")
        print("   Assicurati di aver messo i file nella cartella '_file' (se in locale) o su Drive.")
        return

    items = os.listdir(source_path)
    if not items:
        print(f"⚠️ ATTENZIONE: La cartella sorgente '{source_path}' è vuota.")
        return

    # 3. Iterazione sui contenuti (Dinamica)
    files_processed = 0
    
    for item_name in items:
        source_item = os.path.join(source_path, item_name)
        dest_item = os.path.join(dest_data_dir, item_name)

        try:
            # CASO A: È un file ZIP -> Estrai
            if os.path.isfile(source_item) and item_name.lower().endswith(".zip"):
                print(f"Estrazione ZIP: {item_name}...")
                with zipfile.ZipFile(source_item, 'r') as zip_ref:
                    zip_ref.extractall(dest_data_dir)
                files_processed += 1

            # CASO B: È una Cartella (es. annotations, annotation_json) -> Copia
            elif os.path.isdir(source_item):
                print(f"Copia cartella: {item_name}...")
                # Se esiste già, rimuovila per avere una copia pulita
                if os.path.exists(dest_item):
                    shutil.rmtree(dest_item)
                shutil.copytree(source_item, dest_item)
                files_processed += 1
            
            # (Opzionale) CASO C: Altri file (es. json singoli) -> Copia
            elif os.path.isfile(source_item) and item_name.lower().endswith(".json"):
                 print(f"Copia file: {item_name}...")
                 shutil.copy2(source_item, dest_item)
                 files_processed += 1

        except Exception as e:
            print(f"   ❌ Errore processando '{item_name}': {e}")

    if files_processed > 0:
        print(f"✅ Setup completato! Dati pronti in: {dest_data_dir}")
    else:
        print("⚠️ Nessun file compatibile (zip/cartelle) trovato nella sorgente.")