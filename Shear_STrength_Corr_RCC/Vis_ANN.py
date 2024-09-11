import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

# Load your dataset
data = pd.read_csv("corr_beam.csv")

# Assuming the last column is the output variable
X = data.iloc[:, :-1].values  # Inputs (first 4 columns)
y = data.iloc[:, -1].values   # Output (last column)

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the ANN
model = Sequential()

# Add input layer and hidden layers
model.add(Dense(units=16, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=8, activation='relu'))

# Add output layer (for regression, no activation function or linear activation)
model.add(Dense(units=1))

# Compile the ANN
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on the training data
model.fit(X_train, y_train, epochs=50, batch_size=10)

# Predict the test set results
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)*100
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"r2_Score: {r2:.4f}")


xx = np.linspace(0, 600, 100)
yy = xx
plt.figure()
plt.plot(xx, yy, c='k', linewidth=2)
plt.scatter(y_train, y_pred_train, marker='s')
plt.scatter(y_test, y_pred, marker='o')
plt.grid()
plt.legend(['y=x', 'Training set', 'Testing set'], loc = 'upper left')
plt.tick_params (axis = 'both', which = 'major')
plt.axis('tight')
plt.xlabel('Tested shear strength (kN)')
plt.ylabel('Predicted shear strength (kN)')
plt.xlim([0, 600])
plt.ylim([0, 600])
plt.title("PR Model")
plt.savefig('Vis_ANN_Model.jpeg')
plt.show()