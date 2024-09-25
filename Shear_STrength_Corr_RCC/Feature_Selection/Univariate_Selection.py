import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../corr_beam.csv")
# print(data.head())

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
print("Original Feature Count: ", len(data.columns)-1)
print("Original Feature Count: ", data.iloc[:, :-1].columns)

# Scatter plot of data
x = np.full((158,1), 1)
plt.figure(figsize=(10,10))
plt.scatter(x, data.fc, color="blue")
plt.title("fc")
plt.show()