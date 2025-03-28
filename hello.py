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
import matplotlib
matplotlib.use('TkAgg')  # Use Tk instead of Qt
import matplotlib.pyplot as plt
import os
import shap
import shap
import torch
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

edge_index = torch.load('edge_index.pt')
features = torch.load('features.pt')
labels = torch.load('labels.pt')

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

# Create a directory to save the plots
if not os.path.exists("trust_score_plotsa"):
    os.makedirs("trust_score_plotsa")

# Create a directory to save the models
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

# Training
for epoch in range(180):
    model.train()
    optimizer.zero_grad()

    output = model(features, edge_index)

    loss = F.nll_loss(output, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

        # Save the model
        model_save_path = os.path.join("saved_models", f"trust_gcn_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at epoch {epoch} in: {model_save_path}")

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

        plt.figure()
        plt.bar(bin_centers, num, width=bin_width, align='center')
        plt.xlabel("Trust Scores")
        plt.ylabel("Number of People")
        plt.title(f"Distribution of Trust Scores - Epoch {epoch}")
        plt.savefig(f"trust_score_plotsa/trust_scores_epoch_{epoch}.png")  # Save the plot
        plt.close() # close the plot to prevent overlapping plots.

# Trust Evaluation after final epoch
with torch.no_grad():
    model.eval()
    log_probs = model(features, edge_index)
    influence_scores = torch.exp(log_probs[:, 1])

trust_scores = influence_scores.numpy()

# Plot Trust Distribution final epoch
num, bins = np.histogram(trust_scores, bins=10)
bin_width = bins[1] - bins[0]
bin_centers = (bins[:-1] + bins[1:]) / 2

plt.figure()
plt.bar(bin_centers, num, width=bin_width, align='center')
plt.xlabel("Trust Scores")
plt.ylabel("Number of People")
plt.title(f"Distribution of Trust Scores - Epoch {epoch}")
plt.savefig(f"trust_score_plotsa/trust_scores_epoch_{epoch}.png")  # Save the plot
plt.close()

# Access Control
trust_threshold = 0.65

def grant_access(user_id):
    trust = trust_scores[user_id]
    if trust < trust_threshold:
        return False, "Insufficient trust score"
    else:
        return True, "Access granted"
try:
    df = pd.DataFrame(index=range(len(trust_scores)))
    for user_id in df.index:
        access, message = grant_access(user_id)
        print(f"User {user_id}: {message}")
except NameError:
    print("DataFrame 'df' not defined. Ensure that the 'df' DataFrame is defined before use.")

# ----- Model Prediction for SHAP -----
target_node_index = 1

def model_predict_for_node(x):
    """
    For each row in x (of shape (num_samples, 4)), replace the target node's features
    in the global feature matrix and return the model's prediction for that target node.
    """
    results = []
    for sample in x:
        features_copy = features.copy()
        features_copy[target_node_index] = sample
        features_tensor_local = torch.tensor(features_copy, dtype=torch.float)
        with torch.no_grad():
            pred = model(features_tensor_local, edge_index)[target_node_index]
        results.append(pred)

    # Ensure the output shape is (n_samples, 1)
    return np.array(results).reshape(-1, 1)

# ----- SHAP Explanation -----
background = features[target_node_index].reshape(1, -1)

# KernelExplainer with SHAP
explainer_shap = shap.KernelExplainer(model_predict_for_node, background)
shap_values = explainer_shap.shap_values(features[target_node_index].reshape(1, -1))

# SHAP force plot
print("\nGenerating SHAP force plot for target node...")
shap.initjs()
shap.force_plot(explainer_shap.expected_value, shap_values, features[target_node_index], matplotlib=True, link="logit")
plt.show()

# Extract and display SHAP values
feature_names = ["degree_cent", "betweenness_cent", "closeness_cent", "eigenvector_cent"]

# Display the contributions correctly
print("\nSHAP Values:")
# Flatten the SHAP values if they are in a 2D array format
shap_values_target = np.ravel(shap_values)  # Flatten to 1D

for i, val in enumerate(shap_values_target):
    print(f"{feature_names[i]}: {val:.4f}")


# ----- LIME Explanation -----
def model_predict_for_node_lime(x):
    """
    Ensure the function returns the correct shape for LIME.
    """
    results = []
    for sample in x:
        features_copy = features.copy()
        features_copy[target_node_index] = sample
        features_tensor_local = torch.tensor(features_copy, dtype=torch.float)

        with torch.no_grad():
            pred = model(features_tensor_local, edge_index)[target_node_index].numpy()

        # Ensure prediction shape is consistent with LIME expectations
        if pred.ndim == 0:  # Single value prediction
            results.append([pred])  # Wrap in a list to maintain 2D structure
        else:
            results.append(pred)

    return np.array(results)
