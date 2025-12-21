"""
Hungarian Algorithm Matching per Video Steps e Recipe Embeddings

Questo script esegue il matching ottimale tra:
- Video embeddings (8 steps × 768 features) da hiero_all_video_steps.npz
- Recipe embeddings (N steps × 768 features) da file .pt

Autore: Task2Subtask3
"""

import numpy as np
import torch
import json
import re
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple, List
import argparse


def normalizza_nome_ricetta(nome_ricetta: str) -> str:
    """Normalizza il nome della ricetta per matchare i nomi file .pt"""
    return re.sub(r'[^a-zA-Z0-9]', '', nome_ricetta).lower()


def cosine_similarity_matrix(video_steps: np.ndarray, recipe_steps: np.ndarray) -> np.ndarray:
    """
    Calcola la matrice di similarità coseno tra video steps e recipe steps.
    
    Args:
        video_steps: numpy array shape [8, 768] - embeddings video
        recipe_steps: numpy array shape [N, 768] - embeddings ricetta
    
    Returns:
        numpy array shape [8, N] - matrice di similarità
    """
    # Converti a numpy se necessario
    if isinstance(recipe_steps, torch.Tensor):
        recipe_steps = recipe_steps.numpy()
    
    # Normalizza i vettori
    video_norm = video_steps / (np.linalg.norm(video_steps, axis=1, keepdims=True) + 1e-8)
    recipe_norm = recipe_steps / (np.linalg.norm(recipe_steps, axis=1, keepdims=True) + 1e-8)
    
    # Calcola similarità coseno (matrice 8 x N)
    similarity = np.dot(video_norm, recipe_norm.T)
    
    return similarity


def hungarian_matching(video_steps: np.ndarray, recipe_steps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Esegue il matching ottimale tra video steps e recipe steps usando l'algoritmo ungherese.
    
    Args:
        video_steps: numpy array shape [8, 768]
        recipe_steps: numpy/torch array shape [N, 768]
    
    Returns:
        tuple: (row_indices, col_indices, cost_matrix, total_cost)
    """
    # Calcola matrice di similarità
    similarity = cosine_similarity_matrix(video_steps, recipe_steps)
    
    # Converti similarità in costi (distanza = 1 - similarità)
    cost_matrix = 1 - similarity
    
    # Se le dimensioni sono diverse, creiamo una matrice quadrata con padding
    n_video = video_steps.shape[0]  # 8
    n_recipe = recipe_steps.shape[0] if isinstance(recipe_steps, np.ndarray) else recipe_steps.shape[0]
    
    if n_video != n_recipe:
        # Crea matrice quadrata
        max_dim = max(n_video, n_recipe)
        padded_cost = np.ones((max_dim, max_dim)) * 10  # Alto costo per celle dummy
        padded_cost[:n_video, :n_recipe] = cost_matrix
        cost_matrix_for_hungarian = padded_cost
    else:
        cost_matrix_for_hungarian = cost_matrix
    
    # Applica algoritmo ungherese
    row_ind, col_ind = linear_sum_assignment(cost_matrix_for_hungarian)
    
    # Filtra solo i match validi (non dummy)
    valid_mask = (row_ind < n_video) & (col_ind < n_recipe)
    row_ind = row_ind[valid_mask]
    col_ind = col_ind[valid_mask]
    
    # Calcola costo totale
    total_cost = cost_matrix[row_ind, col_ind].sum()
    
    return row_ind, col_ind, cost_matrix, total_cost


def load_data(data_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """
    Carica tutti i dati necessari.
    
    Returns:
        tuple: (video_features, recipe_embeddings, annotations)
    """
    print("📦 Caricamento dati...")
    
    # Carica video features
    video_path = data_dir / "hiero_all_video_steps.npz"
    video_features = np.load(video_path, allow_pickle=True)
    print(f"✅ Video features: {len(video_features.keys())} video")
    
    # Carica recipe embeddings
    recipe_dir = data_dir / "recipe_embeddings"
    recipe_embeddings = {}
    for pt_file in recipe_dir.glob("*.pt"):
        recipe_name = pt_file.stem
        data = torch.load(pt_file)
        if isinstance(data, dict) and 'embeddings' in data:
            recipe_embeddings[recipe_name] = data['embeddings']
    print(f"✅ Recipe embeddings: {len(recipe_embeddings)} ricette")
    
    # Carica annotazioni
    ann_path = data_dir / "annotation_json" / "complete_step_annotations.json"
    with open(ann_path, 'r') as f:
        annotations = json.load(f)
    print(f"✅ Annotazioni: {len(annotations)} video")
    
    return video_features, recipe_embeddings, annotations


def create_video_to_recipe_map(annotations: Dict) -> Dict[str, str]:
    """Crea mappa video_id -> recipe_name"""
    video_to_recipe = {}
    for video_id, ann_data in annotations.items():
        if 'activity_name' in ann_data:
            activity_name = ann_data['activity_name']
            normalized = normalizza_nome_ricetta(activity_name)
            video_to_recipe[video_id] = normalized
    return video_to_recipe


def process_all_videos(video_features: Dict, recipe_embeddings: Dict, 
                       video_to_recipe: Dict, verbose: bool = True) -> Tuple[Dict, Dict]:
    """
    Processa tutti i video con l'algoritmo ungherese.
    
    Returns:
        tuple: (matching_results, statistics)
    """
    matching_results = {}
    statistics = {
        'total_videos': 0,
        'successful_matches': 0,
        'failed_matches': 0,
        'missing_recipe_embeddings': 0
    }
    
    all_video_ids = list(video_features.keys())
    total_videos = len(all_video_ids)
    
    if verbose:
        print(f"\n🔄 Processamento {total_videos} video...")
        print("="*70)
    
    for i, video_id in enumerate(all_video_ids, 1):
        statistics['total_videos'] += 1
        
        # Progress
        if verbose and (i % 50 == 0 or i == total_videos):
            progress = (i / total_videos) * 100
            print(f"Progress: {i}/{total_videos} ({progress:.1f}%)")
        
        # Ottieni video embeddings
        video_emb = video_features[video_id]
        
        # Trova recipe name
        video_id_base = video_id.replace('_360p_224.mp4_1s_1s', '')
        recipe_name = video_to_recipe.get(video_id_base)
        
        if recipe_name is None:
            statistics['failed_matches'] += 1
            continue
        
        # Ottieni recipe embeddings
        recipe_emb = recipe_embeddings.get(recipe_name)
        
        if recipe_emb is None:
            statistics['missing_recipe_embeddings'] += 1
            continue
        
        # Esegui Hungarian matching
        try:
            row_ind, col_ind, cost_matrix, total_cost = hungarian_matching(video_emb, recipe_emb)
            
            similarity_matrix = 1 - cost_matrix
            avg_similarity = similarity_matrix[row_ind, col_ind].mean()
            
            matching_results[video_id] = {
                'recipe_name': recipe_name,
                'video_steps': row_ind.tolist(),
                'recipe_steps': col_ind.tolist(),
                'total_cost': float(total_cost),
                'avg_similarity': float(avg_similarity),
                'n_video_steps': len(video_emb),
                'n_recipe_steps': len(recipe_emb),
                'matches': list(zip(row_ind.tolist(), col_ind.tolist()))
            }
            
            statistics['successful_matches'] += 1
            
        except Exception as e:
            if verbose:
                print(f"❌ Errore video {video_id}: {str(e)}")
            statistics['failed_matches'] += 1
    
    return matching_results, statistics


def save_results(matching_results: Dict, statistics: Dict, output_dir: Path):
    """Salva i risultati in JSON"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'hungarian_matching_results.json'
    
    results_to_save = {
        'statistics': statistics,
        'matching_results': matching_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\n✅ Risultati salvati in: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1024:.2f} KB")


def print_statistics(statistics: Dict, matching_results: Dict):
    """Stampa statistiche finali"""
    print("\n" + "="*70)
    print("📊 STATISTICHE FINALI:")
    print(f"   Total videos processati: {statistics['total_videos']}")
    print(f"   ✅ Successful matches: {statistics['successful_matches']}")
    print(f"   ⚠️ Failed matches: {statistics['failed_matches']}")
    print(f"   ⚠️ Missing recipe embeddings: {statistics['missing_recipe_embeddings']}")
    
    if matching_results:
        all_similarities = [r['avg_similarity'] for r in matching_results.values()]
        all_costs = [r['total_cost'] for r in matching_results.values()]
        
        print(f"\n   Similarità media: {np.mean(all_similarities):.4f} ± {np.std(all_similarities):.4f}")
        print(f"   Similarità min/max: {np.min(all_similarities):.4f} / {np.max(all_similarities):.4f}")
        print(f"   Costo medio: {np.mean(all_costs):.4f} ± {np.std(all_costs):.4f}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Hungarian Algorithm Matching per Video e Recipe Embeddings')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory contenente i dati (default: data)')
    parser.add_argument('--output-dir', type=str, default='output/hungarian_results',
                       help='Directory di output (default: output/hungarian_results)')
    parser.add_argument('--quiet', action='store_true',
                       help='Modalità silenziosa (meno output)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    verbose = not args.quiet
    
    # Carica dati
    video_features, recipe_embeddings, annotations = load_data(data_dir)
    
    # Crea mappa video -> ricetta
    video_to_recipe = create_video_to_recipe_map(annotations)
    
    # Processa tutti i video
    matching_results, statistics = process_all_videos(
        video_features, recipe_embeddings, video_to_recipe, verbose=verbose
    )
    
    # Salva risultati
    save_results(matching_results, statistics, output_dir)
    
    # Stampa statistiche
    print_statistics(statistics, matching_results)


if __name__ == '__main__':
    main()
