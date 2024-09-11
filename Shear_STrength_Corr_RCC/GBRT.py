import numpy as np
import pandas as pd

dataset = pd.read_csv('corr_beam.csv')
print(dataset.head())

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

y = dataset.loc [:, 'y']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)

# Training the model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
regr_1 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, max_leaf_nodes=5, min_samples_leaf=1, min_samples_split=2, random_state=0, loss='squared_error')
scores = cross_val_score (regr_1, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs = -1)
print('10-fold mean RMSE:', np.mean(np.sqrt( -scores)))


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 

regr_1.fit(X_train, y_train)

# predicting the results
Z1 = regr_1.predict(X_train)
Z2 = regr_1.predict(X_test)
print("GBRT Training R2:", (r2_score(y_train, Z1)*100), "RMSE:", np.sqrt(mean_squared_error(y_train, Z1)), "MAE:", mean_absolute_error(y_train, Z1))
print("GBRT Testing R2:", (r2_score(y_test, Z2)*100), "RMSE:", np.sqrt(mean_squared_error(y_test, Z2)), "MAE:", mean_absolute_error(y_test, Z2))

import matplotlib.pyplot as plt

xx = np.linspace(0, 600, 100)
yy = xx

plt.figure()
plt.plot(xx, yy, c='k', linewidth=2)
plt.scatter(y_train, Z1, marker='s')
plt.scatter(y_test, Z2, marker='o')
plt.grid()
plt.legend(['y=x', 'Training set', 'Testing set'], loc = 'upper left')
plt.tick_params (axis = 'both', which = 'major')
plt.axis('tight')
plt.xlabel('Tested shear strength (kN)')
plt.ylabel('Predicted shear strength (kN)')
plt.xlim([0, 600])
plt.ylim([0, 600])
plt.tight_layout()
plt.savefig("GBRT_Model.jpeg")