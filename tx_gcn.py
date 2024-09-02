import pandas as pd
import networkx as nx
import torch
from torch_geometric.transforms import RandomNodeSplit
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
num_val = int(0.1 * num_nodes)

transform = RandomNodeSplit(split='train_rest', num_splits=1, num_train_per_class=20, num_val=num_val, num_test=0)
data = transform(data)

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

def validate():
    model.eval()
    with torch.no_grad():
        out = model(data)[0]
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()
    return val_loss

best_val_loss = float('inf')
best_model_path = None
num_epochs = 100

for epoch in range(num_epochs):
    train_loss = train()
    val_loss = validate()
    
    model_filename = f'model_epoch_{epoch+1}_val_loss_{val_loss:.4f}.pt'
    torch.save(model.state_dict(), 'gcn_models/' + model_filename)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = model_filename
    
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

print(f'Loading best model from {best_model_path} with Val Loss: {best_val_loss:.4f}')
model.load_state_dict(torch.load(best_model_path))

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