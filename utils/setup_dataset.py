import os
import shutil
import zipfile

def setup_dataset(base_path, data_dir="data"):
    """
    Copia la cartella 'annotation_json' e estrae lo zip 'omnivore_test.zip' nella cartella 'data/'.
    
    Args:
        base_path (str): Percorso dove si trovano 'annotation_json' e 'omnivore_test.zip'.
        data_dir (str, opzionale): Cartella di destinazione. Default: "data".
    """
    # Crea la cartella data se non esiste
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Cartella '{data_dir}' creata.")
    
    # Copia annotation_json
    annotation_src = os.path.join(base_path, "annotation_json")
    annotation_dest = os.path.join(data_dir, "annotation_json")
    
    if not os.path.exists(annotation_src):
        raise FileNotFoundError(f"Cartella '{annotation_src}' non trovata.")
    
    # Se la cartella di destinazione esiste già, la rimuoviamo prima di copiare
    if os.path.exists(annotation_dest):
        shutil.rmtree(annotation_dest)
    
    shutil.copytree(annotation_src, annotation_dest)
    print(f"Cartella 'annotation_json' copiata in '{data_dir}/'.")
    
    # Estrai omnivore_test.zip
    zip_file = os.path.join(base_path, "omnivore_test.zip")
    
    if not os.path.isfile(zip_file):
        raise FileNotFoundError(f"File ZIP '{zip_file}' non trovato.")
    
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    print(f"File ZIP '{zip_file}' estratto in '{data_dir}/'.")

# Esempio d'uso:
# setup_dataset("/content/drive/MyDrive/AML_MistakeDetection_DATA")
