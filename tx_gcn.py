import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

node_transactions_df = pd.read_parquet("data/node_transactions.parquet")
node_transactions_df = node_transactions_df.fillna(0)
edge_transactions_df = pd.read_parquet("data/edge_transactions.parquet")

node_features = torch.tensor(node_transactions_df[['avg_normalized_volume']].values, dtype=torch.float)
node_labels = torch.tensor(node_transactions_df['fraud'].values, dtype=torch.long)
G = nx.from_pandas_edgelist(edge_transactions_df, 'from_id', 'to_id', create_using=nx.DiGraph())
data = from_networkx(G)
node_transactions_df.set_index('node', inplace=True)

data.x = torch.zeros((len(G), node_features.shape[1]), dtype=torch.float)
data.y = torch.zeros(len(G), dtype=torch.long)

node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
for node in node_to_idx:
    if node in node_transactions_df.index:
        idx = node_transactions_df.index.get_loc(node)
        data.x[node_to_idx[node]] = node_features[idx]
        data.y[node_to_idx[node]] = node_labels[idx]

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        embeddings = x
        x = self.classifier(x)
        return x, embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_features = node_features.shape[1]
hidden_dims = 16
model = GCN(num_features=num_features, hidden_dim=hidden_dims).to(device)
data = data.to(device)

num_nodes = data.num_nodes
train_percent = 0.8
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_indices = torch.randperm(num_nodes)[:int(num_nodes * train_percent)]
train_mask[train_indices] = True

data.train_mask = train_mask

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)[0]
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(50):
    loss = train()
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

model.eval()
with torch.no_grad():
    logits, embeddings = model(data)
    embeddings = embeddings.detach().cpu().numpy()
    predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()

embeddings_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
embeddings_df['predicted_class'] = predicted_classes

embeddings_df['actual_class'] = data.y.cpu().numpy()

node_ids = [node for node in G.nodes()]
embeddings_df['node_id'] = node_ids