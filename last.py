import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected, get_laplacian
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt

# 1️⃣ Load and Preprocess Data
df = pd.read_csv("neeww(1).csv")

# Subset for prototyping
sample_size = int(len(df)*.1)
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
        vec1 = feature_matrix[i, 1:4]
        vec2 = feature_matrix[j, 1:4]
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if similarity >= 0.5:
            G.add_edge(i, j)
print(f"Graph created with {len(G.nodes)} nodes and {len(G.edges)} edges.")
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

# Feature matrix with centralities
X = np.array([[centralities['degree'][i],
               centralities['betweenness'][i],
               centralities['closeness'][i],
               centralities['eigenvector'][i]]
              for i in range(num_users)])

X = np.concatenate([feature_matrix[:, [0]], X], axis=1)

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

# Compute Laplacian
laplacian_index, laplacian_weight = get_laplacian(edge_index)

# Features and Labels
features = torch.tensor(X, dtype=torch.float32)
labels = torch.tensor(y, dtype=torch.long)

# 4️⃣ GCN Model with Laplacian
class TrustGCN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=8):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 8)
        self.conv2 = GCNConv(8, 16)
        self.conv3 = GCNConv(16, 8)
        self.conv4 = GCNConv(8, 2)

    def forward(self, x, edge_index, laplacian_index, laplacian_weight):
        # Use Laplacian matrix in propagation
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)

# Model Initialization
model = TrustGCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 5️⃣ Training with SIR Loss
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    output = model(features, edge_index, laplacian_index, laplacian_weight)
    loss = F.nll_loss(output, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 6️⃣ Trust Evaluation
with torch.no_grad():
    model.eval()
    log_probs = model(features, edge_index, laplacian_index, laplacian_weight)
    influence_scores = torch.exp(log_probs[:, 1])

trust_scores = influence_scores.numpy()

# Plot Trust Distribution
num, bins = np.histogram(trust_scores, bins=10)
bin_width = bins[1] - bins[0]
bin_centers = (bins[:-1] + bins[1:]) / 2

plt.bar(bin_centers, num, width=bin_width, align='center')
plt.xlabel("Trust Scores")
plt.ylabel("Number of People")
plt.title("Distribution of Trust Scores")
plt.show()

# Access Control
trust_threshold = 0.65

def grant_access(user_id):
    trust = trust_scores[user_id]
    if trust < trust_threshold:
        return False, "Insufficient trust score"
    else:
        return True, "Access granted"

for user_id in df.index:
    access, message = grant_access(user_id)
    print(f"User {user_id}: {message}")

def calculate_privacy_leakage(granted, user_trusts):
    leakage = 0
    for user, access in enumerate(granted):
        if access and user_trusts[user] < trust_threshold:
            leakage += 1
    return leakage / len(granted)

def calculate_data_integrity(granted, user_trusts):
    integrity = 0
    for user, access in enumerate(granted):
        if access and user_trusts[user] >= trust_threshold:
            integrity += 1
        elif not access:
            integrity += 1
    return integrity / len(granted)


granted = (trust_scores >= trust_threshold).astype(int)

PL = calculate_privacy_leakage(granted, trust_scores)
DI = calculate_data_integrity(granted, trust_scores)

print(f"Privacy Leakage: {PL}, Data Integrity: {DI}")    