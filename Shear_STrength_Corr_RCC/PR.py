import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('corr_beam.csv')
# print(dataset.head())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 1)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)

y_pred_train = regressor.predict(poly_reg.transform(X_train))
y_pred_test = regressor.predict(poly_reg.transform(X_test))

from sklearn.metrics import r2_score
print("r2 Score:  train: ",r2_score(y_train, y_pred_train)*100)
print("r2 Score:  test: ",r2_score(y_test, y_pred_test)*100)

xx = np.linspace(0, 600, 100)
yy = xx

plt.figure()
plt.plot(xx, yy, c='k', linewidth=2)
plt.scatter(y_train, y_pred_train, marker='s')
plt.scatter(y_test, y_pred_test, marker='o')

plt.grid()
plt.legend(['y=x', 'Training set', 'Testing set'], loc = 'upper left')
plt.tick_params (axis = 'both', which = 'major')
plt.axis('tight')
plt.xlabel('Tested shear strength (kN)')
plt.ylabel('Predicted shear strength (kN)')
plt.xlim([0, 600])
plt.ylim([0, 600])
plt.title("PR Model")
plt.savefig('PR_Model.jpeg')
plt.show()
