import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load the dataset
dataset = pd.read_csv('corr_beam.csv')

# Reading the original input variables from the database
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

# Constructing 6 new normalized dimensionless input variables
X = np.zeros(shape=(158, 6))
X[:, 0] = lambda_s
X[:, 1] = h0 / b
X[:, 2] = rho_l * fy / fc
X[:, 3] = rho_v * fyv / fc
X[:, 4] = eta_l
X[:, 5] = eta_w

# Defining the output (shear strength of the corroded beam)
y = dataset.loc[:, 'y']

# Scaling the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# KFold Cross-Validation (10 folds)
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Lists to store R2 scores
r2_scores_train = []
r2_scores_test = []

# Loop through each fold
fold = 1
for train_index, test_index in kf.split(X):
    print(f"\nTraining fold {fold}...")
    
    # Split the data into training and testing sets for each fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize the RandomForestRegressor model for each fold
    regressor = RandomForestRegressor(n_estimators=18, random_state=0)
    regressor.fit(X_train, y_train)
    
    # Predictions on training and testing sets
    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)
    
    # Calculate R2 scores for both train and test sets
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Append R2 scores to the lists
    r2_scores_train.append(r2_train)
    r2_scores_test.append(r2_test)
    
    print(f"Fold {fold} R2 score (train): {r2_train * 100:.2f}")
    print(f"Fold {fold} R2 score (test): {r2_test * 100:.2f}")
    
    fold += 1

# Calculate average R2 scores across the 10 folds
avg_r2_train = np.mean(r2_scores_train) * 100
avg_r2_test = np.mean(r2_scores_test) * 100

print("\nAverage R2 Score across 10 folds (train): ", avg_r2_train)
print("Average R2 Score across 10 folds (test): ", avg_r2_test)

# Optional: Plot the last fold's predictions for visualization

xx = np.linspace(0, 600, 100)
yy = xx

plt.figure()
plt.plot(xx, yy, c='k', linewidth=2)
plt.scatter(y_train, y_pred_train, marker='s', label='Training set')
plt.scatter(y_test, y_pred_test, marker='o', label='Testing set')

plt.grid()
plt.legend(loc='upper left')
plt.tick_params(axis='both', which='major')
plt.axis('tight')
plt.xlabel('Tested shear strength (kN)')
plt.ylabel('Predicted shear strength (kN)')
plt.xlim([0, 600])
plt.ylim([0, 600])
plt.title("RFR Model (Last Fold)")
plt.savefig('RFR_Model_CrossValidation.jpeg')
plt.show()
