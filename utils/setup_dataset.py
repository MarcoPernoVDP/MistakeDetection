import os
import zipfile
import shutil

def setup_dataset(zip_path, data_dir=None):
    """
    Estrae lo ZIP principale in data_dir e organizza:
    - annotation_json/ (tutti i file .json)
    - omnivore/ (estratti tutti i file .npz da omnivore.zip)
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(zip_path), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Cartella temporanea per estrazione dello ZIP principale
    temp_extract = os.path.join(data_dir, "_temp_extract")
    os.makedirs(temp_extract, exist_ok=True)

    # 1️⃣ Estrai lo ZIP principale
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(temp_extract)

    # 2️⃣ Sposta annotation_json/ in data/annotation_json
    ann_src = os.path.join(temp_extract, "annotation_json")
    ann_dst = os.path.join(data_dir, "annotation_json")
    if os.path.exists(ann_dst):
        shutil.rmtree(ann_dst)
    shutil.copytree(ann_src, ann_dst)

    # 3️⃣ Estrai omnivore.zip in data/omnivore
    # Trova il file omnivore.zip
    omnivore_zip = None
    for root, dirs, files in os.walk(temp_extract):
        for f in files:
            if f.lower() == "omnivore.zip":
                omnivore_zip = os.path.join(root, f)
                break
        if omnivore_zip:
            break
    if omnivore_zip is None:
        raise RuntimeError("omnivore.zip non trovato nello ZIP principale.")

    omnivore_dst = os.path.join(data_dir, "omnivore")
    if os.path.exists(omnivore_dst):
        shutil.rmtree(omnivore_dst)
    os.makedirs(omnivore_dst, exist_ok=True)

    # Estrai tutti i file .npz dalla cartella omnivore dentro lo zip
    with zipfile.ZipFile(omnivore_zip, "r") as z:
        for member in z.namelist():
            # Mantieni solo i file .npz e rimuovi eventuali cartelle nidificate
            if member.endswith(".npz"):
                # scrivi direttamente dentro omnivore_dst
                z.extract(member, omnivore_dst)
    
    # Rimuovi cartella temporanea
    shutil.rmtree(temp_extract)

    print(f"Dati organizzati in '{data_dir}':")
    print(f" - annotation_json/ ({len(os.listdir(ann_dst))} file)")
    print(f" - omnivore/ ({len(os.listdir(omnivore_dst))} file)")

    return data_dir
