import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('corr_beam.csv')
# print(dataset.head())

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

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


from sklearn.svm import SVR
regressor = SVR(kernel='linear')
regressor.fit(X, y)

y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

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
plt.title("SVR Model")
plt.savefig('SVR_Model.jpeg')
plt.show()
