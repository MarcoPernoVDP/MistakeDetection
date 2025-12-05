import os
import numpy as np

def inspect_npz_from_dataset(root_dir: str, dataset_folder: str, npz_filename: str, n_rows: int = 5):
    """
    Ispeziona un file .npz partendo dal nome della cartella dataset e dal nome del file.

    Args:
        root_dir (str): directory principale del progetto.
        dataset_folder (str): nome della cartella dentro data/ che contiene i file .npz.
        npz_filename (str): nome del file .npz da ispezionare.
        n_rows (int): numero di righe da mostrare per ogni array.
    """

    npz_path = os.path.join(root_dir, "data", dataset_folder, npz_filename)

    if not os.path.exists(npz_path):
        print(f"[ERROR] File non trovato: {npz_path}")
        return

    data = np.load(npz_path)

    print(f"File: {npz_path}")
    print("Chiavi presenti nel file:", list(data.keys()))

    for key in data.keys():
        arr = data[key]
        print(f"\nArray '{key}' - shape: {arr.shape}, dtype: {arr.dtype}")
        print(arr[:n_rows])
