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

# GCN Model with Batch Normalization
class TrustGCN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = GCNConv(16, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = GCNConv(32, 64)
        self.bn3 = nn.BatchNorm1d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)
    def forward(self, x, edge_index):
        x = F.elu(self.bn1(self.conv1(x, edge_index).relu())) # Apply ReLU after conv
        x = F.elu(self.bn2(self.conv2(x, edge_index).relu()))
        x = F.elu(self.bn3(self.conv3(x, edge_index).relu()))  # Last GCN layer

        # Fully connected processing
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)  # Final output
        return F.log_softmax(x, dim=1)

# Model Initialization with Weight Decay
model = TrustGCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Create directories
if not os.path.exists("trust_score_plotsb"):
    os.makedirs("trust_score_plotsb")
if not os.path.exists("saved_modelss"):
    os.makedirs("saved_modelss")

# Training
for epoch in range(180):
    model.train()
    optimizer.zero_grad()

    output = model(features, edge_index)

    loss = F.nll_loss(output, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

        # Save the model
        model_save_path = os.path.join("saved_modelss", f"trust_gcn_epoch_{epoch}.pth")
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
        plt.savefig(f"trust_score_plotsb/trust_scores_epoch_{epoch}.png")  # Save the plot
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

# ----- Enhanced Model Prediction for SHAP -----
target_node_index = 1
num_background_samples_shap = 100  # Number of background samples to use
feature_names = ["degree_cent", "betweenness_cent", "closeness_cent", "eigenvector_cent"]

def shap_model_predict_proba(x):
    """
    Predicts the probability of the positive class (index 1) for SHAP.
    """
    results = []
    for sample in x:
        features_copy = features.clone().detach()  # Use clone to avoid modifying original
        features_copy[target_node_index] = torch.tensor(sample, dtype=torch.float)
        with torch.no_grad():
            log_probs = model(features_copy, edge_index)
            probs = torch.exp(log_probs[:, 1])  # Probability of class 1
        results.append(probs.item())
    return np.array(results).reshape(-1, 1)

# ----- Enhanced SHAP Explanation -----
# Use a random subset of the features as background data
background_indices = np.random.choice(features.shape[0], size=num_background_samples_shap, replace=False)
background_shap = features[background_indices].numpy()

explainer_shap = shap.KernelExplainer(shap_model_predict_proba, background_shap)
shap_values = explainer_shap.shap_values(features[target_node_index].unsqueeze(0).numpy())

# SHAP force plot
print("\nGenerating SHAP force plot for target node...")
shap.initjs()
shap.force_plot(explainer_shap.expected_value, shap_values, features[target_node_index].numpy(), feature_names=feature_names, matplotlib=True, link="logit")
plt.show()

# Extract and display SHAP values
print("\nSHAP Values for target node (probability of class 1):")
shap_values_target = shap_values[0]  # shap_values is a list for single output

for i, val in enumerate(shap_values_target):
    print(f"{feature_names[i]}: {val:.4f}")

def visualize_shap_summary(all_shap_values, feature_names):
    """
    Visualizes the summary plot of SHAP values.
    """
    plt.figure()
    shap.summary_plot(all_shap_values, features.numpy(), feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot")
    plt.savefig("shap_summary_plot.png")
    plt.close()
    print("\nSHAP Summary Plot saved to shap_summary_plot.png")

# To generate a summary plot (requires SHAP values for multiple instances)
# You would need to adapt the code to calculate SHAP values for a set of nodes
# For demonstration, let's assume we have SHAP values for all nodes (this might be computationally intensive)
# all_node_shap_values = [explainer_shap.shap_values(features[i].unsqueeze(0).numpy()) for i in range(features.shape[0])]
# visualize_shap_summary(np.array([val[0] for val in all_node_shap_values]), feature_names)

# ----- Enhanced Model Prediction for LIME -----
def lime_model_predict_proba(x):
    """
    Predicts the probabilities for both classes for LIME.
    """
    results = []
    for sample in x:
        features_copy = features.clone().detach()
        features_copy[target_node_index] = torch.tensor(sample, dtype=torch.float)
        with torch.no_grad():
            log_probs = model(features_copy, edge_index)
            probs = torch.exp(log_probs)  # Probabilities for both classes
        results.append(probs.numpy())
    return np.array(results)

# ----- Enhanced LIME Explanation -----
num_samples_lime = 1000  # Number of samples to generate for LIME
explainer_lime = LimeTabularExplainer(
    features.numpy(),  # Use the entire feature set as training data for LIME
    mode="classification",
    feature_names=feature_names,
    class_names=["No Trust", "Trust"]  # Provide class names for better interpretation
)

# Explain with LIME
explanation_lime = explainer_lime.explain_instance(
    features[target_node_index].numpy(),
    lime_model_predict_proba,
    num_features=len(feature_names)
)

print("\nLIME Explanation for target node:")
# Display the explanation as a list of (feature, weight) tuples
for feature, weight in explanation_lime.as_list():
    print(f"{feature}: {weight:.4f}")

def visualize_lime_explanation(explanation):
    """
    Visualizes the LIME explanation using matplotlib.
    """
    # explanation.show_in_notebook(show_table=True, show_figure=True) # This might still require a notebook environment

    # Alternative visualization using matplotlib
    feature_weights = explanation.as_list()
    feature_names_exp = [fw[0] for fw in feature_weights]
    weights = [fw[1] for fw in feature_weights]

    plt.figure(figsize=(8, 6))
    plt.barh(feature_names_exp, weights, color=['red' if w < 0 else 'green' for w in weights])
    plt.xlabel("Weight")
    plt.ylabel("Feature")
    plt.title("LIME Explanation for Target Node")
    plt.tight_layout()
    plt.savefig("lime_explanation.png")
    plt.close()
    print("\nLIME Explanation plot saved to lime_explanation.png")

visualize_lime_explanation(explanation_lime)