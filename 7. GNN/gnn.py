import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
data = dataset[0]  # Get the first graph (Cora has only one graph)

# Print dataset information
print(f'Dataset: {dataset}:')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of edges: {data.edge_index.shape[1]}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')

# Define the Graph Convolutional Network model
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Input channels = number of features, output channels = hidden_channels
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        # Input channels = hidden_channels, output channels = number of classes
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        # First Graph Convolution Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second Graph Convolution Layer
        x = self.conv2(x, edge_index)
        
        return x

# Initialize the model
model = GCN(hidden_channels=16)
print(model)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    # Forward pass
    out = model(data.x, data.edge_index)
    # Compute the loss only for the nodes in the training set (using the mask)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    return loss

# Testing function
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    # Use the output of the model to make predictions
    pred = out.argmax(dim=1)
    
    # Calculate accuracy for train, validation and test sets
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum() / data.train_mask.sum()
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum() / data.val_mask.sum()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
    
    return train_acc, val_acc, test_acc

# Train the model
for epoch in range(200):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

# Final evaluation
train_acc, val_acc, test_acc = test()
print(f'Final Train Accuracy: {train_acc:.4f}')
print(f'Final Validation Accuracy: {val_acc:.4f}')
print(f'Final Test Accuracy: {test_acc:.4f}')