import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt

# Load and Preprocess Data
df = pd.read_csv("/home/aryan/sop_fd/neeww(1).csv")

# Subset for prototyping
sample_size = int(len(df)*.05)
sampled_indices = random.sample(range(len(df)), sample_size)
df = df.iloc[sampled_indices].reset_index(drop=True)

# Create the Graph and Add Nodes
G = nx.Graph()
num_users = len(df)
G.add_nodes_from(range(num_users))

# Cosine Similarity Edge Formation
feature_matrix = df.values
for i in range(num_users):
    for j in range(i + 1, num_users):
        vec1 = feature_matrix[i, 0:8]  # Assuming columns 1-3 are features
        vec2 = feature_matrix[j, 0:8]
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if similarity >= 0.5:
            G.add_edge(i, j)

# K-hop Graph Construction
def khop_graph(G, k):
    khop_G = nx.Graph()
    for node in G.nodes:
        neighbors = nx.single_source_shortest_path_length(G, node, cutoff=k)
        for neighbor, dist in neighbors.items():
            if dist <= k:
                khop_G.add_edge(node, neighbor)
    return khop_G

k = 3
G_khop = khop_graph(G, k)
print(f"K-hop graph created with {len(G_khop.nodes)} nodes and {len(G_khop.edges)} edges.")

# Centrality Calculation on K-hop Graph
centralities = {
    'degree': nx.degree_centrality(G_khop),
    'betweenness': nx.betweenness_centrality(G_khop),
    'closeness': nx.closeness_centrality(G_khop),
    'eigenvector': nx.eigenvector_centrality(G_khop, max_iter=1000)
}
print("Centrality calculation completed.")

# Feature matrix with centralities (excluding user ID)
X = np.array([[centralities['degree'][i],
               centralities['betweenness'][i],
               centralities['closeness'][i],
               centralities['eigenvector'][i]]
              for i in range(num_users)])

# SIR Simulation on K-hop Graph
def sir_simulation(G, patient_zero, beta=0.3, gamma=0.1, max_iter=50):
    states = {n: 'S' for n in G.nodes()}
    states[patient_zero] = 'I'
    infected_count = [1]

    for _ in range(max_iter):
        new_infections, recoveries = 0, 0

        for node in list(G.nodes()):
            if states[node] == 'I':
                for neighbor in G.neighbors(node):
                    if states[neighbor] == 'S' and np.random.random() < beta:
                        states[neighbor] = 'I'
                        new_infections += 1

                if np.random.random() < gamma:
                    states[node] = 'R'
                    recoveries += 1

        infected_count.append(infected_count[-1] + new_infections - recoveries)

        if infected_count[-1] == 0:
            break

    return max(infected_count)

sir_labels = np.zeros(num_users)
for i in range(num_users):
    sir_labels[i] = sir_simulation(G_khop, i)
print("SIR simulation completed.")

threshold = np.percentile(sir_labels, 80)
y = (sir_labels >= threshold).astype(int)

# Convert Graph to PyG Edge Index
edge_index = torch.tensor(list(G_khop.edges())).t().contiguous()

# Features and Labels
features = torch.tensor(X, dtype=torch.float32)
labels = torch.tensor(y, dtype=torch.long)

# GCN Model with corrected initialization
class TrustGCN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8):
        super(TrustGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
        
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        
        # Fully connected processing
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Federated Learning Implementation
class FederatedTrustGCN:
    def __init__(self, num_clients=5, input_dim=4, hidden_dim=8):
        self.num_clients = num_clients
        self.global_model = TrustGCN(input_dim, hidden_dim)
        self.client_models = [TrustGCN(input_dim, hidden_dim) for _ in range(num_clients)]
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=0.01) for model in self.client_models]
        
    def distribute_data(self, features, edge_index, labels):
        """Distribute data among clients (simulated partition)"""
        num_users = features.shape[0]
        users_per_client = num_users // self.num_clients
        
        # Create client datasets
        client_data = []
        for i in range(self.num_clients):
            start_idx = i * users_per_client
            end_idx = (i + 1) * users_per_client if i < self.num_clients - 1 else num_users
            
            # Get user indices for this client
            client_indices = list(range(start_idx, end_idx))
            
            # Create subgraph for this client
            client_edge_index = []
            for e in range(edge_index.shape[1]):
                src, dst = edge_index[0, e].item(), edge_index[1, e].item()
                if src in client_indices and dst in client_indices:
                    # Adjust indices to be local to this client
                    local_src = client_indices.index(src)
                    local_dst = client_indices.index(dst)
                    client_edge_index.append([local_src, local_dst])
            
            if not client_edge_index:  # Ensure there's at least one edge
                if len(client_indices) > 1:
                    client_edge_index.append([0, 1])
                    client_edge_index.append([1, 0])
            
            client_edge_index = torch.tensor(client_edge_index).t().contiguous() if client_edge_index else torch.zeros((2, 0), dtype=torch.long)
            client_features = features[client_indices]
            client_labels = labels[client_indices]
            
            client_data.append((client_features, client_edge_index, client_labels, client_indices))
        
        return client_data
    
    def train_client(self, client_id, client_data, epochs=200):
        """Train a client model on its local data"""
        client_features, client_edge_index, client_labels, _ = client_data
        model = self.client_models[client_id]
        optimizer = self.optimizers[client_id]
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(client_features, client_edge_index)
            loss = F.nll_loss(output, client_labels)
            loss.backward()
            optimizer.step()
        
        return loss.item()
    
    def aggregate_models(self, client_weights=None):
        """Aggregate client models using FedAvg algorithm"""
        if client_weights is None:
            client_weights = [1/self.num_clients] * self.num_clients
            
        # Get global model state dict
        global_state = self.global_model.state_dict()
        
        # Initialize with zeros
        for key in global_state:
            global_state[key] = torch.zeros_like(global_state[key])
            
        # Weighted average of client models
        for client_id, weight in enumerate(client_weights):
            client_state = self.client_models[client_id].state_dict()
            for key in global_state:
                global_state[key] += weight * client_state[key]
        
        # Update global model
        self.global_model.load_state_dict(global_state)
        
        # Update client models with global model
        for client_id in range(self.num_clients):
            self.client_models[client_id].load_state_dict(global_state)
    
    def evaluate_global(self, features, edge_index, labels):
        """Evaluate global model performance"""
        self.global_model.eval()
        with torch.no_grad():
            output = self.global_model(features, edge_index)
            loss = F.nll_loss(output, labels)
            pred = output.argmax(dim=1)
            correct = (pred == labels).sum().item()
            accuracy = correct / len(labels)
        return loss.item(), accuracy
    
    def train_federated(self, features, edge_index, labels, rounds=20, client_epochs=100):
        """Run federated learning for multiple rounds"""
        # Distribute data among clients
        client_data = self.distribute_data(features, edge_index, labels)
        
        # Training history
        history = {'loss': [], 'accuracy': []}
        
        print("Starting federated training...")
        for round_num in range(rounds):
            client_losses = []
            
            # Train each client on its local data
            for client_id in range(self.num_clients):
                loss = self.train_client(client_id, client_data[client_id], epochs=client_epochs)
                client_losses.append(loss)
            
            # Aggregate models
            self.aggregate_models()
            
            # Evaluate global model
            loss, accuracy = self.evaluate_global(features, edge_index, labels)
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)
            
            print(f"Round {round_num+1}/{rounds}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return history

# Run federated learning
num_clients = 5  # Number of clients participating in federated learning
fed_gcn = FederatedTrustGCN(num_clients=num_clients)
history = fed_gcn.train_federated(features, edge_index, labels, rounds=20, client_epochs=100)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['loss'])
plt.title('Federated Training Loss')
plt.xlabel('Round')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'])
plt.title('Federated Training Accuracy')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()

# Get trust scores from the global model
with torch.no_grad():
    fed_gcn.global_model.eval()
    log_probs = fed_gcn.global_model(features, edge_index)
    influence_scores = torch.exp(log_probs[:, 1])

trust_scores = influence_scores.numpy()

# Plot Trust Distribution
num, bins = np.histogram(trust_scores, bins=10)
bin_width = bins[1] - bins[0]
bin_centers = (bins[:-1] + bins[1:]) / 2

plt.bar(bin_centers, num, width=bin_width, align='center')
plt.xlabel("Trust Scores")
plt.ylabel("Number of People")
plt.title("Distribution of Trust Scores (Federated Learning)")
plt.show()

# Access Control with Federated Model
trust_threshold = 0.65

def grant_access(user_id):
    trust = trust_scores[user_id]
    # Try to get department info if available
    try:
        department = df.iloc[user_id][4:8].idxmax().split('_')[1]
    except:
        department = "unknown"
    
    if trust < trust_threshold:
        return False, "Insufficient trust score"
    
    if department == "3":  # Assuming department 3 is for medical professionals
        return True, "Full access granted"
    else:
        return True, "Standard access granted"

for user_id in range(min(20, len(df))):  # Show access decisions for first 20 users
    access, message = grant_access(user_id)
    print(f"User {user_id}: {message}")
