# DAGNN Dataset per Mistake Detection

Dataset e DataLoader per DAGNN (Directed Acyclic Graph Neural Network) per la detection di errori nelle ricette di cucina.

## 📋 Panoramica

Il dataset combina:
- **Visual embeddings** (8x768) da video steps tramite EgoVLP
- **Text embeddings** (Nx768) da recipe steps 
- **Hungarian matching** che associa video steps a recipe steps
- **Struttura del grafo** delle ricette (DAG)

## 🏗️ Architettura

### Input Features per ogni Nodo

Ogni nodo del grafo ricetta ha features di dimensione **1536**:
- **Prima metà [0:768]**: Text embedding dello step
- **Seconda metà [768:1536]**: 
  - Visual embedding SE c'è un match dall'Hungarian algorithm
  - **Zeros(768)** SE NON c'è match visuale

### Pattern di Aggiornamento Nodi

```python
# Matching semantico (Hungarian algorithm)
matching = {
    2: 0,  # visual step 2 → recipe node 0 ("taglia cipolla")
    1: 1,  # visual step 1 → recipe node 1 ("accendi gas")
    0: 2,  # visual step 0 → recipe node 2 ("metti pasta")
}

# Aggiornamento features
for visual_idx, recipe_idx in matching.items():
    task_graph.nodes[recipe_idx]["features"] = concat(
        text_embeddings[recipe_idx],    # [768] - text embedding
        visual_steps[visual_idx]         # [768] - visual embedding matchato
    )  # → [1536]

# Per nodi SENZA match:
task_graph.nodes[unmatched_idx]["features"] = concat(
    text_embeddings[unmatched_idx],     # [768] - text embedding
    zeros(768)                           # [768] - nessun match visuale
)  # → [1536]
```

## 📁 Files Creati

### 1. `dataset/dagnn_dataset.py`
Dataset principale con:
- **`DAGNNDataset`**: Classe dataset che combina embeddings e matching
- **`ProjectionLayer`**: Layer allenabile per proiettare [1536] → [256]
- **`collate_fn`**: Funzione per batchare grafi di dimensioni diverse
- **`create_dagnn_dataloader`**: Factory function per creare DataLoader

### 2. `test_dagnn_dataset.py`
Script di test completo che verifica:
- Caricamento dataset
- Batching con DataLoader
- Projection layer
- Demo utilizzo completo

### 3. `example_graph_update.py`
Esempi che mostrano:
- Come aggiornare nodi del grafo con il pattern richiesto
- Utilizzo con dati reali
- Verifica che nodi unmatchati hanno effettivamente zeros

## 🚀 Quick Start

### Installazione
```bash
pip install torch numpy networkx
```

### Uso Base

```python
from dataset.dagnn_dataset import create_dagnn_dataloader, ProjectionLayer

# Crea DataLoader
dataloader = create_dagnn_dataloader(
    video_embeddings_path="data/hiero_all_video_steps.npz",
    recipe_embeddings_dir="data/recipe_embeddings",
    hungarian_results_path="hungarian_results/hungarian_matching_results.json",
    annotation_path="data/annotation_json/error_annotations.json",
    batch_size=8,
    shuffle=True,
)

# Crea projection layer (allenabile)
projection = ProjectionLayer(input_dim=1536, output_dim=256)

# Training loop
for batch in dataloader:
    # 1. Proietta features combinate
    projected_features = []
    for i in range(len(batch['node_features'])):
        # batch['node_features'][i] shape: [N_nodes, 1536]
        proj = projection(batch['node_features'][i])  # → [N_nodes, 256]
        projected_features.append(proj)
    
    # 2. Costruisci grafi (con PyTorch Geometric)
    from torch_geometric.data import Data, Batch
    
    graphs = []
    for i in range(len(batch['node_features'])):
        graph = Data(
            x=projected_features[i],           # [N_nodes, 256]
            edge_index=batch['edge_index'][i], # [2, N_edges]
            y=batch['labels'][i]               # Label (0 o 1)
        )
        graphs.append(graph)
    
    batched_graph = Batch.from_data_list(graphs)
    
    # 3. Forward DAGNN
    output = your_dagnn_model(batched_graph)
    
    # 4. Loss e backward
    loss = criterion(output, batched_graph.y)
    loss.backward()  # Aggiorna sia projection che DAGNN
    optimizer.step()
```

## 📊 Struttura Dati

### Sample dal Dataset

```python
sample = dataset[0]

{
    'node_features': Tensor[N_nodes, 1536],  # Text + Visual embeddings
    'edge_index': Tensor[2, N_edges],        # Graph edges
    'video_key': str,                        # "16_39_360p_224.mp4_1s_1s"
    'recipe_name': str,                      # "scrambledeggs"
    'label': Tensor[1],                      # 0 = no error, 1 = error
    'match_mask': Tensor[N_nodes],           # Bool: True se nodo ha match
    'n_nodes': int,                          # Numero nodi nel grafo
    'metadata': dict,                        # Info aggiuntive (edges, texts, etc)
}
```

### Batch dal DataLoader

```python
batch = next(iter(dataloader))

{
    'node_features': List[Tensor],           # Lista di [N_i, 1536] per ogni grafo
    'edge_index': List[Tensor],              # Lista di [2, E_i] per ogni grafo
    'labels': Tensor[batch_size],            # Labels concatenati
    'match_masks': List[Tensor],             # Lista di [N_i] bool
    'video_keys': List[str],                 # Video identifiers
    'recipe_names': List[str],               # Recipe names
    'n_nodes': List[int],                    # Numero nodi per grafo
    'metadata': List[dict],                  # Metadata per grafo
}
```

## 🧪 Testing

```bash
# Test completo del dataset
python test_dagnn_dataset.py

# Esempi di aggiornamento grafi
python example_graph_update.py
```

Output atteso:
```
Dataset size: 384
Sample 0 Details:
  Video key: 16_39_360p_224.mp4_1s_1s
  Recipe name: scrambledeggs
  Number of nodes: 23
  Node features shape: torch.Size([23, 1536])
  Nodes with visual matches: 8/23
  ✓ Verified: Unmatched nodes have zero visual embeddings
```

## 🔍 Verifica Matching

Il dataset verifica automaticamente che:
- ✅ Nodi con match hanno visual embeddings != 0
- ✅ Nodi senza match hanno visual embeddings == 0
- ✅ Ogni video ha esattamente 8 visual steps
- ✅ Il matching è corretto secondo Hungarian algorithm

```python
# Esempio verifica
sample = dataset[0]
match_mask = sample['match_mask']  # [N_nodes]

for node_idx in range(sample['n_nodes']):
    visual_part = sample['node_features'][node_idx, 768:]  # Ultimi 768
    
    if match_mask[node_idx]:
        assert visual_part.norm() > 0  # Ha visual embedding
    else:
        assert torch.allclose(visual_part, torch.zeros(768))  # Tutti zeros
```

## 💡 Note Importanti

1. **Dimensioni Features**:
   - Input: `[N_nodes, 1536]` (768 text + 768 visual)
   - Dopo ProjectionLayer: `[N_nodes, 256]` (allenabile)

2. **Matching**:
   - Ogni video ha **8 visual steps** (fisso)
   - Ogni ricetta ha **N recipe steps** (variabile)
   - Hungarian matching trova **≤8 match** (alcuni step potrebbero non matchare)

3. **Grafi di Dimensioni Diverse**:
   - Usa `collate_fn` per batchare
   - Ogni grafo nel batch mantiene la sua dimensione
   - PyTorch Geometric `Batch` gestisce grafi variabili

4. **Labels**:
   - `0` = No error
   - `1` = Error detected
   - Estratti da `error_annotations.json`

## 🔧 Personalizzazione

### Cambiare Projection Dimension

```python
# Default: 1536 → 256
projection = ProjectionLayer(input_dim=1536, output_dim=256)

# Custom: 1536 → 512
projection = ProjectionLayer(input_dim=1536, output_dim=512)
```

### Custom Combine Function

```python
# In example_graph_update.py
def custom_combine(text_emb, visual_emb):
    # Weighted concatenation
    alpha = 0.7
    return torch.cat([
        alpha * text_emb,
        (1 - alpha) * visual_emb
    ], dim=-1)

graph = create_task_graph_with_matched_features(
    text_embeddings=text_emb,
    visual_embeddings=visual_emb,
    edge_list=edges,
    matching=matching,
    combine_fn=custom_combine
)
```

## 📈 Statistiche Dataset

```
Total samples: 384
Average nodes per graph: ~17
Average edges per graph: ~23
Average matched nodes: 7-8 / total nodes
Videos with errors: ~50% (varia secondo annotation)
```

## 🎯 Prossimi Passi

1. ✅ Dataset creato e testato
2. ✅ Projection layer implementato
3. ✅ DataLoader configurato
4. ⏳ Implementare DAGNN model
5. ⏳ Training loop completo
6. ⏳ Evaluation metrics

## 📚 Riferimenti

- Video embeddings: EgoVLP + K-means clustering (8 clusters)
- Text embeddings: HiERO text encoder
- Graph structure: Ricetta DAG da metadata
- Matching: Hungarian algorithm (costo minimo)
