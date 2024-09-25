import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import keras
from keras import layers
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('corr_beam.csv')

# Split dataset into features (X) and target (y)
X = dataset.iloc[:, :-1].values  # All columns except the last one (features)
y = dataset.iloc[:, -1].values   # The last column (target - shear strength)

# Feature scaling (Normalization)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the number of folds for cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# Function to create the ANN model
def create_model():
    model = keras.models.Sequential()
    model.add(layers.Dense(units=64, activation='relu', input_shape=(X.shape[1],)))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dense(units=16, activation='relu'))
    model.add(layers.Dense(units=1, activation='linear'))  # 1 output unit for regression
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

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
    
    # Create a new model for each fold
    model = create_model()
    
    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=200, verbose=0)  # Silent training
    
    # Predictions on training and testing sets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
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

# Optionally, plot results from the last fold
xx = np.linspace(0, max(y_test), 100)
yy = xx

plt.figure()
plt.plot(xx, yy, c='k', linewidth=2)
plt.scatter(y_train, y_pred_train, marker='s', color='blue', label="Training set")
plt.scatter(y_test, y_pred_test, marker='o', color='red', label="Testing set")

plt.grid()
plt.legend()
plt.tick_params(axis='both', which='major')
plt.axis('tight')
plt.xlabel('Actual shear strength (kN)')
plt.ylabel('Predicted shear strength (kN)')
plt.xlim([0, max(y_test)])
plt.ylim([0, max(y_test)])
plt.title("ANN Model - Predicted vs Actual Shear Strength (Fold 10)")
plt.show()
