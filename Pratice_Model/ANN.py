import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Shear_Strength_dataset.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras import layers

ann = keras.models.Sequential()
ann.add(layers.Dense(units=17, input_shape=(X_train.shape[1],), activation='relu', kernel_initializer='uniform', name='Hidden_Layer_1'))
ann.add(layers.Dense(units=15, activation='relu', kernel_initializer='uniform', name='Hidden_Layer_2'))
ann.add(layers.Dense(units=13, activation='relu', kernel_initializer='uniform', name='Hidden_Layer_3'))
ann.add(layers.Dense(units=1, activation='linear', kernel_initializer='uniform', name='Output_Layer'))

ann.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

from keras import callbacks
# callback = callbacks.TensorBoard(log_dir='logs', histogram_freq=1)
# ann.fit(X_train, y_train, batch_size=12, verbose=0, epochs=150, callbacks=callback)
ann.fit(X_train, y_train, batch_size=12, verbose=0, epochs=150)
ann.summary()

y_pred = ann.predict(X_test)

# from ann_visualizer.visualize import ann_viz
# ann_viz(ann, view=True, filename='Visualize_ANN.gv', title="ANN Model for Shear Strength.")

# from keras import utils
# utils.plot_model(ann, to_file='Visualize_ANN.png', show_shapes=False)

# from keras import utils

# # Visualize the model and save it to a file
# utils.plot_model(ann, to_file='ANN_Model_Visualization.png', show_shapes=True, show_layer_names=True)


from sklearn.metrics import r2_score
print(f'R2 Score: {r2_score(y_test, y_pred)*100:.2f}%')

ann.save('ANN_Model.h5')
# Method to visulize
# in terminal write pip install netron
# then write netron in terminal
