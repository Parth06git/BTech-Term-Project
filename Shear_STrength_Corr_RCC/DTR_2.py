import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Function for saving max r2 score plot
def isMax(x, lst):
    for i in (lst):
        if(x < i): 
            return False
    return True

# Load the dataset
dataset = pd.read_csv('corr_beam.csv')

# reading the original input variables from the database
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

# constructing 6 new normalized dimensionless input variables
X = np.zeros(shape=(158,6))
X[:, 0] = lambda_s
X[:, 1] = h0/b
X[:, 2] = rho_l * fy / fc
X[:, 3] = rho_v * fyv / fc
X[:, 4] = eta_l
X[:, 5] = eta_w

# defining the output, i.e., the shear strength of the corroded beam
y = dataset.loc [:, 'y']

# Feature scaling (Normalization)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the KFold cross-validator with 10 splits
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Lists to store R2 scores for each fold
r2_scores_train = []
r2_scores_test = []

# Cross-validation loop
fold = 1
for train_index, test_index in kf.split(X):
    print(f"\nTraining fold {fold}...")
    
    # Split the data into training and testing sets for each fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize the Decision Tree Regressor
    regressor = DecisionTreeRegressor(random_state=0)
    
    # Train the model
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
    
    if(isMax(r2_test, r2_scores_test)):
        xx = np.linspace(0, 600, 100)
        yy = xx

        plt.figure()
        plt.plot(xx, yy, c='k', linewidth=2)
        plt.scatter(y_train, y_pred_train, marker='s', color='blue', label="Training set")
        plt.scatter(y_test, y_pred_test, marker='o', color='red', label="Testing set")

        plt.grid()
        plt.legend(loc='upper left')
        plt.tick_params(axis='both', which='major')
        plt.axis('tight')
        plt.xlabel('Tested shear strength (kN)')
        plt.ylabel('Predicted shear strength (kN)')
        plt.xlim([0, 600])
        plt.ylim([0, 600])
        plt.title("DTR Model - Predicted vs Actual Shear Strength")
        plt.savefig('DTR_Model_2.jpeg')
    
    print(f"Fold {fold} R2 score (train): {r2_train * 100:.2f}")
    print(f"Fold {fold} R2 score (test): {r2_test * 100:.2f}")
    
    fold += 1

# Calculate average R2 scores across the 10 folds
avg_r2_train = np.mean(r2_scores_train) * 100
avg_r2_test = np.mean(r2_scores_test) * 100

print("\nAverage R2 Score across 10 folds (train): ", avg_r2_train)
print("Average R2 Score across 10 folds (test): ", avg_r2_test)

# Optionally, you can plot the results from the last fold
xx = np.linspace(0, 600, 100)
yy = xx

plt.figure()
plt.plot(xx, yy, c='k', linewidth=2)
plt.scatter(y_train, y_pred_train, marker='s', color='blue', label="Training set")
plt.scatter(y_test, y_pred_test, marker='o', color='red', label="Testing set")

plt.grid()
plt.legend(loc='upper left')
plt.tick_params(axis='both', which='major')
plt.axis('tight')
plt.xlabel('Tested shear strength (kN)')
plt.ylabel('Predicted shear strength (kN)')
plt.xlim([0, 600])
plt.ylim([0, 600])
plt.title("DTR Model - Predicted vs Actual Shear Strength (Last Fold)")
plt.savefig('DTR_Model_CrossValidation.jpeg')
plt.show()
