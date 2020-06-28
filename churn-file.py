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

print(X.head(5))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

classifier = Sequential()
#Added input layer
classifier.add(Dense(units=10, kernel_initializer = 'he_uniform', activation='relu', input_dim=X_train.shape[1]))
#Adding Hidden layer
classifier.add(Dense(units=6, kernel_initializer = 'he_uniform', activation = 'relu'))
#Adding Output Layer
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation = 'sigmoid'))

classifier.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy'])

model_result = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print("Score is : "+str(score))

# Performing Hyper Parameter Tuning

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

#Commeneted out as it was performed
grid_result = grid.fit(X_train, y_train)

best_score = grid_result.best_score_
best_param = grid_result.best_params_
best_estimator = grid_result.best_estimator_

print("Best Score After Hyper Paramter Tuning : "+str(best_score))
print("Best Parametres are : "+str(best_param))

tuned_classifier = Sequential()

tuned_classifier.add(Dense(units=40, kernel_initializer='he_uniform', activation='relu', input_dim=X_train.shape[1]))
tuned_classifier.add(Dropout(0.3))

tuned_classifier.add(Dense(units=20, kernel_initializer = 'he_uniform', activation='relu'))
tuned_classifier.add(Dropout(0.3))

tuned_classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

tuned_classifier.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy'])

tuned_model_history = tuned_classifier.fit(X_train, y_train, validation_split=0.33, batch_size=128, epochs=40)

tuned_score = accuracy_score(y_pred, y_test)
print("Score after Tuning the model is : "+str(tuned_score))

filename = 'churn-model-85'
import pickle
pickle.dump(model, open(filename, 'wb'))
print("Model saved succesfully!")



