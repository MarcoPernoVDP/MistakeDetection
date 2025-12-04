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

def get_loaders(dataset: Dataset, batch_size: int = 512, val_ratio: float = 0.1, test_ratio: int = 0.2, seed: int = 42):
    # 1. Stampa Info Generali (Shape)
    shape = dataset.shape()
    print("\n" + "="*65)
    print(f"DATASET INFO")
    print(f"   Shape: {shape} -> {shape[0]} Campioni, {shape[1]} Features")
    print("="*65)

    # 2. Split
    total = len(dataset)
    test_len = int(test_ratio * total)
    val_len = int(val_ratio * total)
    train_len = total - test_len - val_len
    
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=gen)
    
    # 3. Stampa Bilanciamento
    print_class_balance(dataset, "FULL DATASET")
    print("-" * 65)
    print_class_balance(train_ds, "TRAIN SET")
    print_class_balance(val_ds, "VALIDATION SET")
    print_class_balance(test_ds, "TEST SET")
    print("="*65 + "\n")
    
    # 4. Crea Loaders
    kwargs = {'num_workers': 0, 'pin_memory': True}
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs)
    )