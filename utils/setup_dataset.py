import os
import shutil
import zipfile

def setup_dataset(base_path, project_root, data_dir_name="data"):
    """
    Copia 'annotation_json' ed estrae zip nella cartella 'data/' nella root del progetto.

    Args:
        base_path (str): Percorso della cartella _file contenente annotation_json e lo zip.
        project_root (str): Percorso della root del progetto.
        data_dir_name (str, opzionale): Nome della cartella dati. Default: "data".
    """
    # Percorso completo della cartella data nella root del progetto
    data_dir = os.path.join(project_root, data_dir_name)
    
    # Crea la cartella data se non esiste
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Cartella '{data_dir}' creata nella root del progetto.")

    # Copia annotation_json
    annotation_src = os.path.join(base_path, "annotation_json")
    annotation_dest = os.path.join(data_dir, "annotation_json")

    if not os.path.exists(annotation_src):
        raise FileNotFoundError(f"Cartella '{annotation_src}' non trovata.")

    if os.path.exists(annotation_dest):
        shutil.rmtree(annotation_dest)

    shutil.copytree(annotation_src, annotation_dest)
    print(f"Cartella 'annotation_json' copiata in '{data_dir}/'.")

    # Trova e estrai eventuali zip nella cartella _file
    for file_name in os.listdir(base_path):
        if file_name.endswith(".zip"):
            zip_path = os.path.join(base_path, file_name)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"File ZIP '{zip_path}' estratto in '{data_dir}/'.")
