"""
Example showing how to update graph node features with matched embeddings.

This demonstrates the exact pattern you described:
1. Get matching from Hungarian algorithm
2. Update task graph nodes with visual embeddings
3. Use zeros for unmatched nodes
"""

import torch
import networkx as nx
from typing import Dict


def create_task_graph_with_matched_features(
    text_embeddings: torch.Tensor,
    visual_embeddings: torch.Tensor,
    edge_list: list,
    matching: Dict[int, int],
    combine_fn=None,
):
    """
    Create a task graph and update node features with matched visual embeddings.
    
    This implements the pattern:
    ```
    matching = {
        2: 0,  # visual step 2 → task node 0
        1: 1,  # visual step 1 → task node 1
        0: 2,  # visual step 0 → task node 2
    }
    
    for visual_idx, task_idx in matching.items():
        task_graph.nodes[task_idx]["features"] = combine(
            task_graph.nodes[task_idx]["features"],  # text embedding
            visual_steps[visual_idx]                  # visual embedding
        )
    ```
    
    Args:
        text_embeddings: [N_nodes, 768] - Text embeddings for each recipe step
        visual_embeddings: [8, 768] - Visual embeddings from video steps
        edge_list: List of [source, target] edges
        matching: Dict mapping visual_idx -> task_idx (recipe node index)
        combine_fn: Function to combine text and visual embeddings
                   If None, uses concatenation
    
    Returns:
        NetworkX DiGraph with updated node features
    """
    if combine_fn is None:
        # Default: concatenate text + visual
        def combine_fn(text_emb, visual_emb):
            return torch.cat([text_emb, visual_emb], dim=-1)
    
    n_nodes = text_embeddings.shape[0]
    embedding_dim = text_embeddings.shape[1]  # 768
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes with initial text features
    for node_idx in range(n_nodes):
        G.add_node(node_idx, text_features=text_embeddings[node_idx])
    
    # Add edges
    for src, tgt in edge_list:
        G.add_edge(src, tgt)
    
    # Create reverse mapping: task_idx -> visual_idx
    task_to_visual = {task_idx: visual_idx for visual_idx, task_idx in matching.items()}
    
    # Update node features based on matching
    zero_visual = torch.zeros(embedding_dim)  # Use zeros for unmatched nodes
    
    for task_idx in range(n_nodes):
        text_emb = text_embeddings[task_idx]
        
        if task_idx in task_to_visual:
            # Node has a visual match
            visual_idx = task_to_visual[task_idx]
            visual_emb = visual_embeddings[visual_idx]
        else:
            # Node has no visual match - use zeros
            visual_emb = zero_visual
        
        # Combine text and visual embeddings
        combined_features = combine_fn(text_emb, visual_emb)
        
        # Update node features
        G.nodes[task_idx]["features"] = combined_features
    
    return G


def example_usage():
    """Example showing the exact pattern you described."""
    print("=" * 80)
    print("Example: Updating Task Graph with Matched Visual Embeddings")
    print("=" * 80)
    
    # Simulate recipe with 5 steps
    n_recipe_steps = 5
    text_embeddings = torch.randn(n_recipe_steps, 768)
    
    # Simulate video with 8 visual steps
    visual_embeddings = torch.randn(8, 768)
    
    # Define recipe graph structure
    edge_list = [
        [0, 1],  # step 0 → step 1
        [1, 2],  # step 1 → step 2
        [2, 3],  # step 2 → step 3
        [3, 4],  # step 3 → step 4
    ]
    
    # Hungarian matching results (visual_idx → recipe_idx)
    # Only 3 out of 5 recipe steps have visual matches
    matching = {
        2: 0,  # visual step 2 → recipe step 0 ("taglia cipolla")
        1: 1,  # visual step 1 → recipe step 1 ("accendi gas")
        0: 2,  # visual step 0 → recipe step 2 ("metti pasta")
    }
    # Recipe steps 3 and 4 have NO visual match (will get zeros)
    
    print(f"\nRecipe steps: {n_recipe_steps}")
    print(f"Visual steps: {visual_embeddings.shape[0]}")
    print(f"Matched steps: {len(matching)}")
    print(f"Unmatched recipe steps: {n_recipe_steps - len(matching)}")
    
    print(f"\nMatching (visual_idx → recipe_idx):")
    for visual_idx, recipe_idx in matching.items():
        print(f"  Visual step {visual_idx} → Recipe step {recipe_idx}")
    
    # Create graph with updated features
    task_graph = create_task_graph_with_matched_features(
        text_embeddings=text_embeddings,
        visual_embeddings=visual_embeddings,
        edge_list=edge_list,
        matching=matching,
    )
    
    print(f"\n" + "=" * 80)
    print("Task Graph Node Features:")
    print("=" * 80)
    
    # Reverse mapping for display
    task_to_visual = {task_idx: visual_idx for visual_idx, task_idx in matching.items()}
    
    for node_idx in task_graph.nodes():
        features = task_graph.nodes[node_idx]["features"]
        text_part = features[:768]
        visual_part = features[768:]
        
        has_visual = node_idx in task_to_visual
        visual_info = f"matched to visual step {task_to_visual[node_idx]}" if has_visual else "NO MATCH (zeros)"
        
        print(f"\nRecipe Step {node_idx}:")
        print(f"  Status: {visual_info}")
        print(f"  Combined features shape: {features.shape}")
        print(f"  Text part (first 768): norm = {text_part.norm():.4f}")
        print(f"  Visual part (last 768): norm = {visual_part.norm():.4f}")
        
        if not has_visual:
            # Verify it's actually zeros
            assert torch.allclose(visual_part, torch.zeros_like(visual_part))
            print(f"  ✓ Verified: Visual part is all zeros")
    
    print(f"\n" + "=" * 80)
    print("Graph Structure:")
    print("=" * 80)
    print(f"Nodes: {list(task_graph.nodes())}")
    print(f"Edges: {list(task_graph.edges())}")
    
    return task_graph


def example_with_dagnn_dataset():
    """Example using real data from DAGNN Dataset."""
    print("\n\n" + "=" * 80)
    print("Example: Using Real Data from DAGNN Dataset")
    print("=" * 80)
    
    from dataset.dagnn_dataset import DAGNNDataset
    
    # Load dataset
    dataset = DAGNNDataset(
        video_embeddings_path="data/hiero_all_video_steps.npz",
        recipe_embeddings_dir="data/recipe_embeddings",
        hungarian_results_path="hungarian_results/hungarian_matching_results.json",
        annotation_path="data/annotation_json/error_annotations.json",
    )
    
    # Get first sample
    sample = dataset[0]
    
    print(f"\nVideo: {sample['video_key']}")
    print(f"Recipe: {sample['recipe_name']}")
    print(f"Number of nodes: {sample['n_nodes']}")
    
    # Extract text and visual parts from combined features
    combined = sample['node_features']  # [N_nodes, 1536]
    text_part = combined[:, :768]
    visual_part = combined[:, 768:]
    
    # Check which nodes have matches
    match_mask = sample['match_mask']
    
    print(f"\nNode-by-node analysis:")
    for node_idx in range(sample['n_nodes']):
        has_match = match_mask[node_idx].item()
        visual_norm = visual_part[node_idx].norm().item()
        
        status = "MATCHED" if has_match else "UNMATCHED (zeros)"
        print(f"  Node {node_idx:2d}: {status:20s} | visual norm = {visual_norm:8.4f}")
        
        # Verify unmatched nodes have zero visual embeddings
        if not has_match:
            assert torch.allclose(visual_part[node_idx], torch.zeros(768), atol=1e-6)
    
    # Build NetworkX graph
    edges = sample['metadata']['edges']
    G = nx.DiGraph()
    
    for node_idx in range(sample['n_nodes']):
        G.add_node(
            node_idx,
            features=combined[node_idx],
            text_features=text_part[node_idx],
            visual_features=visual_part[node_idx],
            has_match=match_mask[node_idx].item(),
        )
    
    for src, tgt in edges:
        G.add_edge(src, tgt)
    
    print(f"\nGraph info:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Matched nodes: {match_mask.sum().item()}/{sample['n_nodes']}")
    
    # Example: Access node features
    print(f"\nExample node (node 0):")
    print(f"  Has match: {G.nodes[0]['has_match']}")
    print(f"  Features shape: {G.nodes[0]['features'].shape}")
    
    return G


def demonstration_pseudocode():
    """Print the exact pseudocode you described."""
    print("\n\n" + "=" * 80)
    print("Your Original Pseudocode Pattern:")
    print("=" * 80)
    print("""
# Matching trova le corrispondenze semantiche
matching = {
    2: 0,  # visual step 2 → task node 0 ("taglia cipolla")
    1: 1,  # visual step 1 → task node 1 ("accendi gas")
    0: 2,  # visual step 0 → task node 2 ("metti pasta")
}

# Aggiorna le features dei nodi (NON gli indici!)
for visual_idx, task_idx in matching.items():
    task_graph.nodes[task_idx]["features"] = combine(
        task_graph.nodes[task_idx]["features"],  # text embedding
        visual_steps[visual_idx]                  # visual embedding matchato
    )
    
# OR if no match:
# task_graph.nodes[task_idx]["features"] = combine(
#     task_graph.nodes[task_idx]["features"],
#     zeros
# )
    """)
    
    print("\n" + "=" * 80)
    print("How This is Implemented in DAGNNDataset:")
    print("=" * 80)
    print("""
In dagnn_dataset.py:

1. Load text embeddings: [N_nodes, 768]
2. Load visual embeddings: [8, 768]
3. Load matching: {visual_idx: recipe_idx}

4. For each node:
   - Get text embedding
   - If node has match in matching dict:
       → Get corresponding visual embedding
   - Else:
       → Use zeros(768)
   - Combine: [text_emb || visual_emb] → [N_nodes, 1536]

5. Return combined features ready for DAGNN

6. ProjectionLayer can then transform [1536] → [256] (learnable)
    """)


if __name__ == "__main__":
    # Run examples
    task_graph = example_usage()
    
    real_graph = example_with_dagnn_dataset()
    
    demonstration_pseudocode()
    
    print("\n" + "=" * 80)
    print("✅ Examples completed!")
    print("=" * 80)
