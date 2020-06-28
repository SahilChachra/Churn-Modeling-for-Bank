import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./Churn_Modelling.csv')
print(data.head())

X = data.iloc[:, 3:13]
y = data.iloc[:, 13]
geo = pd.get_dummies(X['Geography'],drop_first=True)
gen = pd.get_dummies(X['Gender'], drop_first=True)
Geography = geo.copy()
Gender = gen.copy()
X = X.drop(['Gender','Geography'], axis = 1)
X = pd.concat([X,Gender, Geography],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid
from keras.wrappers.scikit_learn import KerasClassifier

def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
    model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    model.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy'])
    return model

from sklearn.model_selection import GridSearchCV
model = KerasClassifier(build_fn=create_model)

batches = [128,256]
epochs = [30]
layers = [[20], [40,20], [45,30,15]]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation = activations, batch_size=batches, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid)

import time
t0 = time.time()
grid_result = grid.fit(X_train, y_train)
#t1=time.time
#print(t1-t0)

print(grid_result.best_score_)
print(grid_result.best_params_)
