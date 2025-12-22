"""
Dataset for DAGNN (Directed Acyclic Graph Neural Network) for cooking mistake detection.

This dataset combines:
- Visual embeddings from video steps (8x768) from EgoVLP
- Text embeddings from recipe steps (Nx768) 
- Hungarian matching results that map video_steps to recipe_steps
- Recipe graph structure (edges, nodes)
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import networkx as nx


class DAGNNDataset(Dataset):
    """
    Dataset that creates graph structures for DAGNN training.
    
    Each sample contains:
    - A directed acyclic graph (DAG) representing the recipe
    - Node features: combination of text embeddings + matched visual embeddings (or zeros)
    - Graph structure from recipe metadata
    - Labels for error detection
    """
    
    def __init__(
        self,
        video_embeddings_path: str,
        recipe_embeddings_dir: str,
        hungarian_results_path: str,
        annotation_path: str,
        projection_dim: Optional[int] = None,
        use_projection: bool = True,
    ):
        """
        Args:
            video_embeddings_path: Path to hiero_all_video_steps.npz
            recipe_embeddings_dir: Path to directory containing recipe .pt files
            hungarian_results_path: Path to hungarian_matching_results.json
            annotation_path: Path to error annotations (for labels)
            projection_dim: If specified, embeddings will be projected to this dimension
            use_projection: Whether to use learnable projection (will be added in model)
        """
        self.video_embeddings_path = video_embeddings_path
        self.recipe_embeddings_dir = recipe_embeddings_dir
        self.hungarian_results_path = hungarian_results_path
        self.annotation_path = annotation_path
        self.projection_dim = projection_dim
        self.use_projection = use_projection
        
        # Load data
        print("Loading video embeddings...")
        self.video_embeddings = np.load(video_embeddings_path)
        
        print("Loading Hungarian matching results...")
        with open(hungarian_results_path, 'r') as f:
            hungarian_data = json.load(f)
            self.matching_results = hungarian_data['matching_results']
        
        print("Loading error annotations...")
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Create list of valid samples (videos that have matching + recipe embeddings)
        self.samples = []
        self._build_sample_list()
        
        print(f"Dataset initialized with {len(self.samples)} samples")
    
    def _build_sample_list(self):
        """Build list of valid samples."""
        for video_key in self.matching_results.keys():
            match_info = self.matching_results[video_key]
            recipe_name = match_info.get('recipe_name')
            
            if recipe_name is None:
                continue
            
            # Check if recipe embeddings exist
            recipe_path = os.path.join(self.recipe_embeddings_dir, f"{recipe_name}.pt")
            if not os.path.exists(recipe_path):
                continue
            
            # Check if video embeddings exist
            if video_key not in self.video_embeddings:
                continue
            
            self.samples.append({
                'video_key': video_key,
                'recipe_name': recipe_name,
                'recipe_path': recipe_path
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict:
        """
        Returns a dictionary containing:
        - node_features: Combined text + visual embeddings [N_nodes, 768*2 or projection_dim]
        - edge_index: Graph edges [2, N_edges]
        - video_key: Video identifier
        - recipe_name: Recipe name
        - label: Error label (if available)
        - match_mask: Binary mask indicating which nodes have visual matches [N_nodes]
        """
        sample_info = self.samples[idx]
        video_key = sample_info['video_key']
        recipe_name = sample_info['recipe_name']
        recipe_path = sample_info['recipe_path']

        if video_key == "13_5_360p_224.mp4_1s_1s":
            print("Debugging sample:", video_key, recipe_name)
        
        # Load visual embeddings (8x768)
        visual_embeddings = torch.from_numpy(
            self.video_embeddings[video_key]
        ).float()  # [8, 768]
        
        # Load recipe embeddings and metadata
        recipe_data = torch.load(recipe_path, weights_only=False)
        text_embeddings = recipe_data['embeddings']  # [N_steps, 768]
        metadata = recipe_data['metadata']
        
        # Get matching information
        match_info = self.matching_results[video_key]
        matches = match_info.get('matches', [])  # List of [video_idx, recipe_idx] pairs
        
        # Create matching dictionary: recipe_idx -> video_idx
        recipe_to_visual = {}
        for video_idx, recipe_idx in matches:
            recipe_to_visual[recipe_idx] = video_idx
        
        # Combine embeddings: text + matched visual (or zeros)
        n_nodes = text_embeddings.shape[0]
        embedding_dim = text_embeddings.shape[1]  # 768
        
        # Initialize combined features [N_nodes, 768*2]
        combined_features = torch.zeros(n_nodes, embedding_dim * 2)
        match_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        for node_idx in range(n_nodes):
            # Add text embedding
            combined_features[node_idx, :embedding_dim] = text_embeddings[node_idx]
            
            # Add visual embedding if match exists
            if node_idx in recipe_to_visual:
                visual_idx = recipe_to_visual[node_idx]
                combined_features[node_idx, embedding_dim:] = visual_embeddings[visual_idx]
                match_mask[node_idx] = True
            # else: keep zeros for visual part
        
        # Build edge index from metadata
        edges = metadata.get('edges', [])
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t()  # [2, N_edges]
        else:
            # If no edges, create empty tensor
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Get label (error detection)
        label = self._get_label(video_key)
        
        return {
            'node_features': combined_features,  # [N_nodes, 768*2]
            'edge_index': edge_index,  # [2, N_edges]
            'video_key': video_key,
            'recipe_name': recipe_name,
            'label': label,
            'match_mask': match_mask,  # [N_nodes] - which nodes have visual matches
            'n_nodes': n_nodes,
            'metadata': metadata,  # Keep for additional info if needed
        }
    
    def _get_label(self, video_key: str) -> torch.Tensor:
        """
        Extract error label from annotations.
        
        Returns:
            Binary label: 1 if error exists, 0 otherwise
        """
        # Extract recording_id from video_key (format: "activityID_recordingID_...")
        parts = video_key.split('_')
        if len(parts) >= 2:
            recording_id = f"{parts[0]}_{parts[1]}"
        else:
            recording_id = video_key
        
        # Check if this recording has errors
        if recording_id in self.annotations:
            has_error = self.annotations[recording_id]['has_errors']
            return torch.tensor(1 if has_error else 0, dtype=torch.long)
        
        # Default: no error
        return torch.tensor(0, dtype=torch.long)


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching graphs with different sizes.
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched dictionary with:
        - node_features: List of [N_i, feature_dim] tensors
        - edge_index: List of [2, E_i] tensors
        - labels: [batch_size] tensor
        - match_masks: List of [N_i] tensors
        - Other metadata
    """
    return {
        'node_features': [item['node_features'] for item in batch],
        'edge_index': [item['edge_index'] for item in batch],
        'labels': torch.stack([item['label'] for item in batch]),
        'match_masks': [item['match_mask'] for item in batch],
        'video_keys': [item['video_key'] for item in batch],
        'recipe_names': [item['recipe_name'] for item in batch],
        'n_nodes': [item['n_nodes'] for item in batch],
        'metadata': [item['metadata'] for item in batch],
    }


def create_dagnn_dataloader(
    video_embeddings_path: str,
    recipe_embeddings_dir: str,
    hungarian_results_path: str,
    annotation_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
):
    """
    Convenience function to create DataLoader for DAGNN.
    
    Args:
        video_embeddings_path: Path to hiero_all_video_steps.npz
        recipe_embeddings_dir: Path to recipe embeddings directory
        hungarian_results_path: Path to Hungarian matching results
        annotation_path: Path to error annotations
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        **kwargs: Additional arguments for Dataset or DataLoader
    
    Returns:
        torch.utils.data.DataLoader
    """
    dataset = DAGNNDataset(
        video_embeddings_path=video_embeddings_path,
        recipe_embeddings_dir=recipe_embeddings_dir,
        hungarian_results_path=hungarian_results_path,
        annotation_path=annotation_path,
        **{k: v for k, v in kwargs.items() if k in ['projection_dim', 'use_projection']}
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **{k: v for k, v in kwargs.items() if k not in ['projection_dim', 'use_projection']}
    )
    
    return dataloader


# Example usage pseudocode for DAGNN model:
"""
# In your DAGNN forward pass:

def update_graph_features(self, batch, projection_layer):
    '''
    Update graph node features with combined text + visual embeddings.
    
    Args:
        batch: Batch from DAGNNDataset
        projection_layer: ProjectionLayer instance (learnable)
    '''
    updated_graphs = []
    
    for i in range(len(batch['node_features'])):
        # Get combined features [N_nodes, 1536]
        combined_features = batch['node_features'][i]
        
        # Apply learnable projection [N_nodes, 1536] -> [N_nodes, 256]
        projected_features = projection_layer(combined_features)
        
        # Create graph with updated features
        edge_index = batch['edge_index'][i]
        
        # Build graph (using PyTorch Geometric or DGL)
        graph = create_graph(
            num_nodes=batch['n_nodes'][i],
            edge_index=edge_index,
            node_features=projected_features
        )
        
        updated_graphs.append(graph)
    
    return updated_graphs

# Usage in training loop:
projection = ProjectionLayer(input_dim=1536, output_dim=256)
dagnn_model = YourDAGNNModel(...)

for batch in dataloader:
    # Update graph features with projection
    graphs = update_graph_features(batch, projection)
    
    # Forward pass through DAGNN
    outputs = dagnn_model(graphs)
    
    # Compute loss and backprop
    loss = criterion(outputs, batch['labels'])
    loss.backward()  # This updates both projection and DAGNN weights
"""
