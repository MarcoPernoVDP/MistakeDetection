from collections import defaultdict
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset

# --- UTILS INTEGATE ---

def print_class_balance(dataset: Dataset, name: str = "Dataset"):
    if isinstance(dataset, Subset):
        labels = dataset.dataset.y[dataset.indices]
    else:
        labels = dataset.y
    
    cnt_0 = (labels == 0).sum().item()
    cnt_1 = (labels == 1).sum().item()
    total = cnt_0 + cnt_1
    
    print(f"{name:<18} | Tot: {total:<6} | OK: {cnt_0:<5} ({cnt_0/total:.1%}) | ERR: {cnt_1:<5} ({cnt_1/total:.1%})", end="")
    if cnt_1 > 0: print(f" | Ratio: 1:{cnt_0/cnt_1:.1f}")
    else: print("")

def print_class_balance_v2(dataset: Dataset, name: str = "Dataset"):
    """Stampa il bilanciamento per V2 (conteggio a livello di step e sottosecondi)"""
    
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        step_indices = dataset.indices
    else:
        base_dataset = dataset
        step_indices = list(range(len(dataset)))
    
    # Numero di step nel subset
    num_steps = len(step_indices)
    
    # Raccogliere tutti i sotto-secondi di questi step
    all_subsec_labels = []
    
    for step_idx in step_indices:
        # Ottieni lo step_id dalla lista di unique_step_ids
        step_id = base_dataset.unique_step_ids[step_idx]
        
        # Trova tutti i sotto-secondi che appartengono a questo step
        import numpy as np
        mask = base_dataset.step_ids == step_id
        subsec_labels = base_dataset.y[mask]
        all_subsec_labels.append(subsec_labels)
    
    # Concatena tutti i label
    if all_subsec_labels:
        y = torch.cat(all_subsec_labels)
    else:
        y = torch.tensor([])
    
    # Conteggio
    cnt_0 = (y == 0).sum().item()
    cnt_1 = (y == 1).sum().item()
    total_subsecs = cnt_0 + cnt_1
    
    if total_subsecs == 0:
        print(f"{name:<18} | Steps: {num_steps:<4} | Subsecs: 0      | OK: 0     (N/A) | ERR: 0     (N/A)")
    else:
        print(f"{name:<18} | Steps: {num_steps:<4} | Subsecs: {total_subsecs:<5} | OK: {cnt_0:<5} ({cnt_0/total_subsecs:.1%}) | ERR: {cnt_1:<5} ({cnt_1/total_subsecs:.1%})", end="")
        if cnt_1 > 0: print(f" | Ratio: 1:{cnt_0/cnt_1:.1f}")
        else: print("")

def custom_collate_fn(batch):
    """
    Custom collate per gestire:
    - V1: batch di (X, y) singoli
    - V2: batch di sequenze (X_seq, y_seq, step_id, seq_len)
           senza padding (batch_size sempre 1)
    """
    if len(batch[0]) == 2:  # V1
        X, y = zip(*batch)
        return torch.stack(X), torch.stack(y)
    
    else:  # V2 - batch_size = 1, niente padding
        X_seq, y_seq, step_id, seq_len = batch[0]
        
        return {
            'X': X_seq,                    # shape: (seq_len, 1024)
            'y': y_seq,                    # shape: (seq_len,)
            'step_id': step_id,            # stringa es: "10_3_5"
            'seq_len': seq_len             # scalare
        }

def get_mlp_loaders(dataset: Dataset, batch_size: int = 512, val_ratio: float = 0.1, test_ratio: float = 0.2, seed: int = 42):
    # 1. Stampa Info Generali
    print("\n" + "="*85)
    shape = dataset.shape()
    print(f"DATASET INFO [V1 - SUBSECOND-BASED]")
    print(f"   Shape: {shape} -> {shape[0]} Campioni, {shape[1]} Features")
    print("="*85)

    # 2. Split
    total = len(dataset)
    test_len = int(test_ratio * total)
    val_len = int(val_ratio * total)
    train_len = total - test_len - val_len
    
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = split_by_step(dataset, [train_len, val_len, test_len], generator=gen)
    
    # 3. Stampa Bilanciamento
    print_class_balance(dataset, "FULL DATASET")
    print("-" * 85)
    print_class_balance(train_ds, "TRAIN SET")
    print_class_balance(val_ds, "VALIDATION SET")
    print_class_balance(test_ds, "TEST SET")
    
    print("="*85 + "\n")
    
    # 4. Crea Loaders (con custom collate per V2)
    kwargs = {'num_workers': 0, 'pin_memory': True, 'collate_fn': custom_collate_fn}
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs)
    )

def get_tranformer_loaders(dataset: Dataset, batch_size: int = 512, val_ratio: float = 0.1, test_ratio: float = 0.2, seed: int = 42):
    # 1. Stampa Info Generali
    print("\n" + "="*85)
    print(f"DATASET INFO [V2 - STEP-BASED]")
    print(f"   Total Steps: {len(dataset)}")
    print(f"   Total Sub-seconds: {len(dataset.X)}")
    print(f"   Avg seconds per step: {len(dataset.X) / len(dataset):.2f}")

    # 2. Split
    total = len(dataset)
    test_len = int(test_ratio * total)
    val_len = int(val_ratio * total)
    train_len = total - test_len - val_len
    
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=gen)
    
    # 3. Stampa Bilanciamento
    print_class_balance_v2(dataset, "FULL DATASET")
    print("-" * 85)
    print_class_balance_v2(train_ds, "TRAIN SET")
    print_class_balance_v2(val_ds, "VALIDATION SET")
    print_class_balance_v2(test_ds, "TEST SET")
    
    print("="*85 + "\n")
    
    # 4. Crea Loaders (con custom collate per V2)
    kwargs = {'num_workers': 0, 'pin_memory': True, 'collate_fn': custom_collate_fn}
    
    # V2: batch_size 1 per avere uno step per batch
    batch_size = 1
    print(f"[V2] Batch size forzato a 1 (uno step per batch)")
    print(f"[V2] Training loop itererà su {len(train_ds)} step\n")
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs)
    )

def split_by_step(dataset: Dataset, lengths: list[int], generator: torch.Generator):
    """
    Divide il dataset in train/val/test garantendo che tutti gli elementi
    con lo stesso (video, step) rimangano insieme nello stesso split.
    """

    # ---- 1. Raggruppa per (video, step) ----
    groups = defaultdict(list)

    for idx in range(len(dataset)):
        X, y, step_id, video_id = dataset[idx]   # ← struttura del tuo dataset

        groups[(video_id, step_id)].append(idx)

    # ---- 2. Shuffle dei gruppi ----
    group_keys = list(groups.keys())
    perm = torch.randperm(len(group_keys), generator=generator)
    group_keys = [group_keys[i] for i in perm]

    # ---- 3. Assegnazione ai tre split ----
    train_len, val_len, test_len = lengths
    train_idx, val_idx, test_idx = [], [], []

    count_train = count_val = count_test = 0

    for key in group_keys:
        idxs = groups[key]
        group_size = len(idxs)

        if count_train + group_size <= train_len:
            train_idx.extend(idxs)
            count_train += group_size

        elif count_val + group_size <= val_len:
            val_idx.extend(idxs)
            count_val += group_size

        else:
            test_idx.extend(idxs)
            count_test += group_size

    # ---- 4. Restituisce i Subset ----
    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )

def split_by_video():
    pass