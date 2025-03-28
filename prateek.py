import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random
import matplotlib.pyplot as plt

# Loading the data 
df = pd.read_csv(r"/home/aryan/sop_fd/neeww(1).csv")

# Using the full dataset
sample_size = len(df)*.1
df = df.sample(n=int(sample_size), random_state=42)
df = df.reset_index(drop=True)

# Creating the graph and adding nodes 
G = nx.Graph()
num_users = len(df)
G.add_nodes_from(range(num_users))

# Adding the cosine similarity here
feature_matrix = df.values
for i in range(num_users):
    for j in range(i+1, num_users):
        user_i_features = feature_matrix[i, 1:4]
        user_j_features = feature_matrix[j, 1:4]
        
        # Proper cosine similarity calculation
        similarity = np.dot(user_i_features, user_j_features) / (
            np.linalg.norm(user_i_features) * np.linalg.norm(user_j_features) + 1e-8)
        
        if similarity >= 0.5:  # Using 0.5 as threshold mentioned in paper
            G.add_edge(i, j)

# Calculate centralities
centralities = {
    'degree': nx.degree_centrality(G),
    'betweenness': nx.betweenness_centrality(G),
    'closeness': nx.closeness_centrality(G),
    'eigenvector': nx.eigenvector_centrality(G, max_iter=500)
}
print("centrality done")

X = np.array([[centralities['degree'][i],
               centralities['betweenness'][i],
               centralities['closeness'][i],
               centralities['eigenvector'][i]]
              for i in range(num_users)])

X = np.concatenate([feature_matrix[:, [0]], X], axis=1)

# SIR Simulation for Ground Truth Labels
def sir_simulation(G, patient_zero, beta=0.3, gamma=0.1, max_iter=50):
    states = {n: 'S' for n in G.nodes()}
    states[patient_zero] = 'I'
    infected_count = [1]

    for _ in range(max_iter):
        new_infections = 0
        recoveries = 0

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
    sir_labels[i] = sir_simulation(G, i)
print("sir done")

threshold = np.percentile(sir_labels, 80)
y = (sir_labels >= threshold).astype(int)

class TrustGCN(nn.Module):
    def __init__(self, input_dim):
        super(TrustGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, 8)
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

edge_index = torch.tensor(list(G.edges())).t().contiguous()
features = torch.tensor(X, dtype=torch.float32)
labels = torch.tensor(y, dtype=torch.long)

# Training with SIR-based Loss
model = TrustGCN(input_dim=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Fading factor implementation
def apply_fading_factor(G, pl, di):
    rho = (1 - pl + di) / 2
    for u, v in G.edges():
        G[u][v]['weight'] *= rho
    return G

# Training loop
num_epochs = 200
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    
    # Mini-batch training
    permutation = torch.randperm(features.size()[0])
    for i in range(0, features.size()[0], batch_size):
        optimizer.zero_grad()
        
        indices = permutation[i:i+batch_size]
        batch_features = features[indices]
        batch_labels = labels[indices]
        
        output = model(batch_features, edge_index)
        loss = F.nll_loss(output, batch_labels)
        
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Trust Evaluation and Access Control
with torch.no_grad():
    model.eval()
    log_probs = model(features, edge_index)
    influence_scores = torch.exp(log_probs[:, 1])

trust_scores = influence_scores.numpy()

# Visualize trust score distribution
plt.figure(figsize=(10, 6))
num, bins, _ = plt.hist(trust_scores, bins=20, edgecolor='black')
plt.xlabel("Trust Scores")
plt.ylabel("Number of Users")
plt.title("Distribution of Trust Scores")
plt.show()

# Apply fading factor
privacy_leakage = np.random.random(num_users)
data_integrity = np.random.random(num_users)
G = apply_fading_factor(G, privacy_leakage, data_integrity)

# Access control function
def grant_access(user_id, trust_threshold=0.65):
    department = df.iloc[user_id][4:8].idxmax().split('_')[1]
    trust = trust_scores[user_id]

    if trust < trust_threshold:
        return False, "Insufficient trust score"

    if department == '3':
        return True, "Full access granted"
    else:
        return True, "Standard access granted"

# Apply access control to all users
for user_id in df.index:
    access, message = grant_access(user_id)
    print(f"User {user_id}: {message}")
