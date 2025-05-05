import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Step 1: Download the dataset from Kaggle
def download_kaggle_dataset(dataset_name, dataset_path):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path=dataset_path, unzip=True)

dataset_name = "snap/stanford-cora"  # Example Kaggle dataset (Cora dataset)
dataset_path = "./data"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    download_kaggle_dataset(dataset_name, dataset_path)

# Step 2: Load dataset (assuming Cora format)
node_data = pd.read_csv(os.path.join(dataset_path, "cora.content"), sep='\t', header=None)
edge_data = pd.read_csv(os.path.join(dataset_path, "cora.cites"), sep='\t', header=None)

# Extract features and labels
features = torch.tensor(node_data.iloc[:, 1:-1].values, dtype=torch.float)
labels = torch.tensor(pd.factorize(node_data.iloc[:, -1])[0], dtype=torch.long)

# Create edge index tensor
edge_index = torch.tensor(edge_data.values.T, dtype=torch.long)

# Step 3: Create PyTorch Geometric Data object
graph_data = Data(x=features, edge_index=edge_index, y=labels)

# Step 4: Define a basic GNN Model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Step 5: Model Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(graph_data.x.shape[1], 16, len(labels.unique())).to(device)
graph_data = graph_data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()

# Train model
def train():
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    loss = criterion(out, graph_data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Train for 200 epochs
for epoch in range(200):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Step 6: Evaluate Model
def test():
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred == graph_data.y).sum()
        acc = int(correct) / int(graph_data.y.size(0))
        print(f'Accuracy: {acc:.4f}')

test()