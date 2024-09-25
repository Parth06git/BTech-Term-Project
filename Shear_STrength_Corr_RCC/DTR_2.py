import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score

# Load the dataset
dataset = pd.read_csv('corr_beam.csv')

# Split dataset into features and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Eliminate rows where the target (y) is greater than 300
mask = y <= 300
X = X[mask]
y = y[mask]

# Feature scaling
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Feature Selection using Recursive Feature Elimination with Cross-Validation (RFECV)
regressor = DecisionTreeRegressor(random_state=0)
rfecv = RFECV(estimator=regressor, cv=5, scoring='r2')  # 5-fold cross-validation
X_rfe = rfecv.fit_transform(X, y)

# Optimal number of features
print(f"Optimal number of features: {rfecv.n_features_}")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.2, random_state=1)

# Train the model on selected features
regressor.fit(X_train, y_train)

# Predictions on training and testing data
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

# Evaluate the model performance
print("r2 Score:  train: ", r2_score(y_train, y_pred_train) * 100)
print("r2 Score:  test: ", r2_score(y_test, y_pred_test) * 100)

# Plot the results
xx = np.linspace(0, 300, 100)  # Adjust the range since y values are <= 300
yy = xx

plt.figure()
plt.plot(xx, yy, c='k', linewidth=2)
plt.scatter(y_train, y_pred_train, marker='s')
plt.scatter(y_test, y_pred_test, marker='o')

plt.grid()
plt.legend(['y=x', 'Training set', 'Testing set'], loc='upper left')
plt.tick_params(axis='both', which='major')
plt.axis('tight')
plt.xlabel('Tested shear strength (kN)')
plt.ylabel('Predicted shear strength (kN)')
plt.xlim([0, 300])  # Adjust the plot range to fit the data
plt.ylim([0, 300])
plt.title("DTR Model 2")
plt.savefig('DTR_Model_2.jpeg')
plt.show()
