"""
Complete example using DAGNN Dataset with PyTorch Geometric.

This shows the full pipeline:
1. Load data with DAGNNDataset
2. Project features with learnable layer
3. Create PyG graphs
4. Simple GNN model
5. Training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from dataset.dagnn_dataset import create_dagnn_dataloader, ProjectionLayer


class SimpleDAGNN(nn.Module):
    """
    Simple DAGNN model for binary classification (error detection).
    
    Architecture:
    1. ProjectionLayer: [1536] → [256]
    2. GCN layers on the graph
    3. Global pooling
    4. Binary classifier
    """
    
    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Feature projection
        self.projection = ProjectionLayer(
            input_dim=input_dim,
            output_dim=hidden_dim
        )
        
        # GNN layers
        self.convs = nn.ModuleList()
        for i in range(num_gnn_layers):
            self.convs.append(
                GCNConv(hidden_dim, hidden_dim)
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # Binary classification
        )
    
    def forward(self, batch_data):
        """
        Args:
            batch_data: Batched PyG Data object with:
                - x: [total_nodes, 1536] node features
                - edge_index: [2, total_edges] edges
                - batch: [total_nodes] batch assignment
        
        Returns:
            [batch_size, 2] - logits for binary classification
        """
        x = batch_data.x
        edge_index = batch_data.edge_index
        batch = batch_data.batch
        
        # Project features
        x = self.projection(x)  # [total_nodes, 1536] → [total_nodes, 256]
        
        # GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling (one embedding per graph)
        x = global_mean_pool(x, batch)  # [batch_size, 256]
        
        # Classification
        out = self.classifier(x)  # [batch_size, 2]
        
        return out


def collate_to_pyg(batch_dict):
    """
    Convert batch from DAGNNDataset to PyTorch Geometric format.
    
    Args:
        batch_dict: Dictionary from DAGNN collate_fn
    
    Returns:
        Batched PyG Data object
    """
    graphs = []
    
    for i in range(len(batch_dict['node_features'])):
        graph = Data(
            x=batch_dict['node_features'][i],        # [N_i, 1536]
            edge_index=batch_dict['edge_index'][i],  # [2, E_i]
            y=batch_dict['labels'][i],               # Scalar
        )
        graphs.append(graph)
    
    # Batch graphs
    batched = Batch.from_data_list(graphs)
    
    return batched


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Convert to PyG format
        pyg_batch = collate_to_pyg(batch).to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(pyg_batch)
        
        # Loss
        loss = criterion(outputs, pyg_batch.y)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += pyg_batch.y.size(0)
        correct += predicted.eq(pyg_batch.y).sum().item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: "
                  f"Loss = {loss.item():.4f}, "
                  f"Acc = {100.*correct/total:.2f}%")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        # Convert to PyG format
        pyg_batch = collate_to_pyg(batch).to(device)
        
        # Forward
        outputs = model(pyg_batch)
        loss = criterion(outputs, pyg_batch.y)
        
        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += pyg_batch.y.size(0)
        correct += predicted.eq(pyg_batch.y).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    """Main training script."""
    print("=" * 80)
    print("DAGNN Training with PyTorch Geometric")
    print("=" * 80)
    
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create DataLoader
    print("\nCreating DataLoader...")
    dataloader = create_dagnn_dataloader(
        video_embeddings_path="data/hiero_all_video_steps.npz",
        recipe_embeddings_dir="data/recipe_embeddings",
        hungarian_results_path="hungarian_results/hungarian_matching_results.json",
        annotation_path="data/annotation_json/error_annotations.json",
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )
    
    print(f"Total batches: {len(dataloader)}")
    
    # Create model
    print("\nCreating model...")
    model = SimpleDAGNN(
        input_dim=1536,
        hidden_dim=256,
        num_gnn_layers=3,
        dropout=0.1,
    ).to(device)
    
    print(f"Model: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print("\n" + "=" * 80)
    print("Training (Demo - 3 epochs)")
    print("=" * 80)
    
    num_epochs = 3
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, dataloader, optimizer, criterion, device
        )
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Acc:  {train_acc:.2f}%")
    
    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print("=" * 80)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'dagnn_checkpoint.pth')
    
    print("\nModel saved to: dagnn_checkpoint.pth")


def demo_single_batch():
    """Demo processing a single batch."""
    print("\n" + "=" * 80)
    print("Demo: Processing Single Batch")
    print("=" * 80)
    
    # Create DataLoader
    dataloader = create_dagnn_dataloader(
        video_embeddings_path="data/hiero_all_video_steps.npz",
        recipe_embeddings_dir="data/recipe_embeddings",
        hungarian_results_path="hungarian_results/hungarian_matching_results.json",
        annotation_path="data/annotation_json/error_annotations.json",
        batch_size=4,
        shuffle=False,
    )
    
    # Get batch
    batch = next(iter(dataloader))
    
    print(f"\nBatch from DAGNNDataset:")
    print(f"  Type: {type(batch)}")
    print(f"  Keys: {batch.keys()}")
    print(f"  Batch size: {len(batch['node_features'])}")
    
    # Convert to PyG
    pyg_batch = collate_to_pyg(batch)
    
    print(f"\nConverted to PyTorch Geometric:")
    print(f"  Type: {type(pyg_batch)}")
    print(f"  Num nodes: {pyg_batch.num_nodes}")
    print(f"  Num edges: {pyg_batch.num_edges}")
    print(f"  Num graphs: {pyg_batch.num_graphs}")
    print(f"  Node features shape: {pyg_batch.x.shape}")
    print(f"  Edge index shape: {pyg_batch.edge_index.shape}")
    print(f"  Labels shape: {pyg_batch.y.shape}")
    print(f"  Batch assignment: {pyg_batch.batch.shape}")
    
    # Create model
    model = SimpleDAGNN(input_dim=1536, hidden_dim=256)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(pyg_batch)
    
    print(f"\nModel outputs:")
    print(f"  Shape: {outputs.shape}")  # [batch_size, 2]
    print(f"  Logits: {outputs}")
    
    # Predictions
    probs = F.softmax(outputs, dim=1)
    preds = outputs.argmax(dim=1)
    
    print(f"\nPredictions:")
    for i in range(len(batch['node_features'])):
        print(f"  Graph {i} ({batch['recipe_names'][i]}):")
        print(f"    True label: {batch['labels'][i].item()}")
        print(f"    Predicted:  {preds[i].item()}")
        print(f"    Prob(error): {probs[i, 1].item():.4f}")


if __name__ == "__main__":
    # Demo single batch first
    demo_single_batch()
    
    # Then run training
    try:
        print("\n" + "=" * 80)
        print("Ready to train? This will run 3 epochs.")
        print("=" * 80)
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
