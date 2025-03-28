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
sample_size = int(len(df)*.5)
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

# GCN Model
class TrustGCN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8):   # input_dim=4 as only X(feature matrix(centrality dim=num_users*4 is passed not social attrbutes that was used to make edge_index so not passed)
        super().__init__()
        self.conv1 = GCNConv(input_dim, 16) #users*4->users*16
        self.conv2 = GCNConv(16, 32)    #users*16->users*32
        self.conv3 = GCNConv(32, 64)   #users*32->users*64
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)  # Final output layer (users*2)  2 is req becuase edge_index is2*users so match 
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))  # Last GCN layer
        
        # Fully connected processing
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)  # Final output
        return F.log_softmax(x, dim=1)

# Model Initialization
model = TrustGCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Training
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    
    output = model(features, edge_index)
    
    loss = F.nll_loss(output, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Trust Evaluation
with torch.no_grad():
    model.eval()
    log_probs = model(features, edge_index)
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