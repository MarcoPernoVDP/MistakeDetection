"""
Example usage and testing script for DAGNN Dataset.

This script demonstrates how to:
1. Load the DAGNN dataset
2. Inspect a sample
3. Create a DataLoader
4. Use with a projection layer
"""

import torch
from dataset.dagnn_dataset import (
    DAGNNDataset, 
    ProjectionLayer, 
    create_dagnn_dataloader,
    collate_fn
)


def test_dataset():
    """Test basic dataset functionality."""
    print("=" * 80)
    print("Testing DAGNN Dataset")
    print("=" * 80)
    
    # Paths
    video_embeddings_path = "data/hiero_all_video_steps.npz"
    recipe_embeddings_dir = "data/recipe_embeddings"
    hungarian_results_path = "hungarian_results/hungarian_matching_results.json"
    annotation_path = "data/annotation_json/error_annotations.json"
    
    # Create dataset
    dataset = DAGNNDataset(
        video_embeddings_path=video_embeddings_path,
        recipe_embeddings_dir=recipe_embeddings_dir,
        hungarian_results_path=hungarian_results_path,
        annotation_path=annotation_path,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    
    print("\n" + "=" * 80)
    print("Sample 0 Details:")
    print("=" * 80)
    print(f"Video key: {sample['video_key']}")
    print(f"Recipe name: {sample['recipe_name']}")
    print(f"Number of nodes: {sample['n_nodes']}")
    print(f"Node features shape: {sample['node_features'].shape}")
    print(f"Edge index shape: {sample['edge_index'].shape}")
    print(f"Label: {sample['label'].item()}")
    print(f"Match mask: {sample['match_mask']}")
    print(f"Nodes with visual matches: {sample['match_mask'].sum().item()}/{sample['n_nodes']}")
    
    # Show which nodes have matches
    matched_indices = torch.where(sample['match_mask'])[0].tolist()
    print(f"Matched node indices: {matched_indices}")
    
    return dataset


def test_dataloader():
    """Test DataLoader with batching."""
    print("\n" + "=" * 80)
    print("Testing DataLoader")
    print("=" * 80)
    
    # Create DataLoader
    dataloader = create_dagnn_dataloader(
        video_embeddings_path="data/hiero_all_video_steps.npz",
        recipe_embeddings_dir="data/recipe_embeddings",
        hungarian_results_path="hungarian_results/hungarian_matching_results.json",
        annotation_path="data/annotation_json/error_annotations.json",
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"Number of batches: {len(dataloader)}")
    
    # Get first batch
    batch = next(iter(dataloader))
    
    print("\nBatch details:")
    print(f"Batch size: {len(batch['node_features'])}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Labels: {batch['labels']}")
    
    for i in range(len(batch['node_features'])):
        print(f"\nGraph {i}:")
        print(f"  Video: {batch['video_keys'][i]}")
        print(f"  Recipe: {batch['recipe_names'][i]}")
        print(f"  Nodes: {batch['n_nodes'][i]}")
        print(f"  Node features: {batch['node_features'][i].shape}")
        print(f"  Edges: {batch['edge_index'][i].shape}")
        print(f"  Matches: {batch['match_masks'][i].sum()}/{batch['n_nodes'][i]}")
    
    return dataloader, batch


def test_projection():
    """Test projection layer."""
    print("\n" + "=" * 80)
    print("Testing Projection Layer")
    print("=" * 80)
    
    # Create projection layer
    projection = ProjectionLayer(input_dim=1536, output_dim=256)
    
    print(f"Projection layer: {projection}")
    print(f"Input dim: 1536 (768 text + 768 visual)")
    print(f"Output dim: 256")
    
    # Test with dummy data
    dummy_features = torch.randn(10, 1536)  # 10 nodes, 1536 features
    projected = projection(dummy_features)
    
    print(f"\nTest projection:")
    print(f"  Input shape: {dummy_features.shape}")
    print(f"  Output shape: {projected.shape}")
    
    return projection


def demo_usage_with_projection(dataloader, projection):
    """Demonstrate how to use dataset with projection in training loop."""
    print("\n" + "=" * 80)
    print("Demo: Using Dataset with Projection in Training Loop")
    print("=" * 80)
    
    # Get a batch
    batch = next(iter(dataloader))
    
    print("\nProcessing batch through projection layer...")
    
    projected_features_list = []
    for i in range(len(batch['node_features'])):
        # Get combined features [N_nodes, 1536]
        combined = batch['node_features'][i]
        
        # Apply projection [N_nodes, 1536] -> [N_nodes, 256]
        projected = projection(combined)
        
        projected_features_list.append(projected)
        
        print(f"\nGraph {i} ({batch['recipe_names'][i]}):")
        print(f"  Original features: {combined.shape}")
        print(f"  Projected features: {projected.shape}")
        print(f"  Edge index: {batch['edge_index'][i].shape}")
        
        # Pseudocode for graph construction
        print(f"  → Ready for DAGNN with {batch['n_nodes'][i]} nodes")
    
    print("\n" + "=" * 80)
    print("Pseudocode for DAGNN usage:")
    print("=" * 80)
    print("""
    # In your training loop:
    for batch in dataloader:
        # 1. Project combined features
        graphs = []
        for i in range(len(batch['node_features'])):
            projected_features = projection_layer(batch['node_features'][i])
            
            # 2. Create graph object (e.g., with PyTorch Geometric)
            graph = Data(
                x=projected_features,           # [N_nodes, 256]
                edge_index=batch['edge_index'][i],  # [2, N_edges]
                y=batch['labels'][i]            # Scalar label
            )
            graphs.append(graph)
        
        # 3. Batch graphs
        batched_graph = Batch.from_data_list(graphs)
        
        # 4. Forward through DAGNN
        output = dagnn_model(batched_graph)
        
        # 5. Compute loss
        loss = criterion(output, batched_graph.y)
        
        # 6. Backward (updates both projection and DAGNN)
        loss.backward()
        optimizer.step()
    """)


def main():
    """Run all tests."""
    print("DAGNN Dataset Testing Suite")
    print("=" * 80)
    
    # Test 1: Dataset
    dataset = test_dataset()
    
    # Test 2: DataLoader
    dataloader, batch = test_dataloader()
    
    # Test 3: Projection
    projection = test_projection()
    
    # Test 4: Complete usage demo
    demo_usage_with_projection(dataloader, projection)
    
    print("\n" + "=" * 80)
    print("✅ All tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
