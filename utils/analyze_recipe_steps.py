import json
from collections import defaultdict
import statistics

def analyze_recipe_steps(json_file_path):
    """
    Analizza il file JSON per calcolare statistiche sugli steps per ricetta.
    
    Args:
        json_file_path: Path al file JSON da analizzare
    """
    # Carica il file JSON
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Dizionario per raggruppare le entries per ricetta
    recipe_entries = defaultdict(list)
    
    # Analizza ogni chiave nel formato xx_yy
    for key in data.keys():
        # Estrae l'ID della ricetta (la parte prima dell'underscore)
        recipe_id = key.split('_')[0]
        
        # Conta gli steps per questa entry
        num_steps = len(data[key]['steps'])
        
        # Aggiunge il numero di steps alla lista per questa ricetta
        recipe_entries[recipe_id].append(num_steps)
    
    # Stampa i risultati
    print("=" * 70)
    print("ANALISI STEPS PER RICETTA")
    print("=" * 70)
    print(f"\nNumero totale di ricette: {len(recipe_entries)}")
    print(f"Numero totale di entries nel file: {len(data)}")
    print(f"\n{'=' * 70}")
    print(f"{'Ricetta':<10} {'Entries':<10} {'Min':<10} {'Max':<10} {'Media':<10}")
    print(f"{'=' * 70}")
    
    # Calcola e stampa le statistiche per ogni ricetta
    results = {}
    for recipe_id in sorted(recipe_entries.keys(), key=int):
        steps_list = recipe_entries[recipe_id]
        min_steps = min(steps_list)
        max_steps = max(steps_list)
        avg_steps = statistics.mean(steps_list)
        num_entries = len(steps_list)
        
        print(f"{recipe_id:<10} {num_entries:<10} {min_steps:<10} {max_steps:<10} {avg_steps:<10.2f}")
        
        results[recipe_id] = {
            'num_entries': num_entries,
            'min_steps': min_steps,
            'max_steps': max_steps,
            'avg_steps': avg_steps
        }
    
    print("=" * 70)
    
    # Salva i risultati in un file JSON
    output_file = "recipe_steps_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n✓ Risultati salvati in: {output_file}")
    
    return results


if __name__ == "__main__":
    # Path al file JSON
    json_file = "data/annotation_json/complete_step_annotations.json"
    
    # Esegui l'analisi
    results = analyze_recipe_steps(json_file)
