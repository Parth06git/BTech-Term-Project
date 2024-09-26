import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
dataset = pd.read_csv('corr_beam.csv')

# Reading the original input variables
fc = dataset.loc[:, 'fc']
b = dataset.loc[:, 'b']
h = dataset.loc[:, 'h']
rho_l = dataset.loc[:, 'rho_l']
rho_v = dataset.loc[:, 'rho_v']
fy = dataset.loc[:, 'fy']
fyv = dataset.loc[:, 'fyv']
s = dataset.loc[:, 's']
lambda_s = dataset.loc[:, 'lambda_s']
eta_l = dataset.loc[:, 'eta_l']
eta_w = dataset.loc[:, 'eta_w']
h0 = dataset.loc[:, 'h0']

# Constructing normalized dimensionless input variables
X = np.zeros(shape=(158, 6))
X[:, 0] = lambda_s
X[:, 1] = h0 / b
X[:, 2] = rho_l * fy / fc
X[:, 3] = rho_v * fyv / fc
X[:, 4] = eta_l
X[:, 5] = eta_w

# Defining output (shear strength of the corroded beam)
y = dataset.loc[:, 'y']

# Feature names for labeling
feature_names = [
    "λ_s (lambda_s)", 
    "h0 / b", 
    "ρ_l * fy / fc", 
    "ρ_v * fyv / fc", 
    "η_l (eta_l)", 
    "η_w (eta_w)"
]

# Create a figure to hold subplots
plt.figure(figsize=(12, 8))

# Loop through each feature in X to create box and scatter plots
for i in range(6):
    plt.subplot(2, 3, i + 1)  # Create a 2x3 grid of subplots
    
    # Boxplot for the feature
    plt.boxplot(X[:, i], vert=False, patch_artist=False, widths=0.4)
    
    # Scatter plot overlaying the boxplot
    plt.scatter(X[:, i], np.ones_like(X[:, i]), alpha=0.7, color='blue', edgecolors='black')
    
    # Labeling the x-axis with feature names and the y-axis as "y"
    plt.xlabel(feature_names[i])
    plt.ylabel("Shear Strength (y)")
    plt.title(f"Scatter Plot and Box Plot for {feature_names[i]}")

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the combined plot
plt.savefig("Feature_Scatter_Plot")
plt.show()
