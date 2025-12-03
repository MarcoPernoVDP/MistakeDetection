import os
import shutil
import zipfile

def setup_dataset(base_path, project_root, data_dir_name="data"):
    """
    Copia 'annotation_json' ed estrae zip nella cartella 'data/' nella root del progetto.
    Funziona anche su Google Drive / Colab.
    """
    # Cartella data nella root del progetto
    data_dir = os.path.join(project_root, data_dir_name)
    os.makedirs(data_dir, exist_ok=True)

    # Copia annotation_json
    annotation_src = os.path.join(base_path, "annotation_json")
    annotation_dest = os.path.join(data_dir, "annotation_json")

    if not os.path.exists(annotation_src):
        raise FileNotFoundError(f"Cartella '{annotation_src}' non trovata.")

    if os.path.exists(annotation_dest):
        shutil.rmtree(annotation_dest)

    shutil.copytree(annotation_src, annotation_dest)
    print(f"Cartella 'annotation_json' copiata in '{data_dir}/'.")

    # Estrai solo file .zip presenti nella cartella base_path
    for file_name in os.listdir(base_path):
        zip_path = os.path.join(base_path, file_name)
        if os.path.isfile(zip_path) and file_name.endswith(".zip"):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"File ZIP '{zip_path}' estratto in '{data_dir}/'.")
