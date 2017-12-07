import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
import keras
import time

from model_building import get_test_data, get_training_data, save_submission
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def create_neural_network(optimizer='adam',drop_out=True, units=60):
    input_dim=8
    model = Sequential()
    model.add(Dense(units=units, kernel_initializer='normal', activation='relu', input_dim=input_dim))
    if drop_out==True: model.add(Dropout(0.2))
    # model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    # # model.add(Dense(32, kernel_initializer='normal', activation='relu', input_dim=input_dim))
    # if drop_out==True: model.add(Dropout(0.25))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    print 'get data'
    X, y = get_training_data()
    X_testing = get_test_data()

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    IDs = X_testing.index.values
    X_testing = X_testing.reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    X_testing = np.asarray(X_testing)

    print 'build neural network'
    # # model = create_neural_network()
    # nn = KerasClassifier(build_fn=create_neural_network, verbose=2)
    # params_grid = {'optimizer':['rmsprop','adagrad'],
    #                 'epochs': [100,200],
    #                 'batch_size':[5,10],
    #                 'drop_out': [True],
    #                 'units': [20,50]}
    #
    # kfold = StratifiedKFold(n_splits=20, shuffle=True)
    # print 'grid search'
    # grid = GridSearchCV(estimator=nn, param_grid=params_grid,cv=kfold, n_jobs=3)
    # grid_results = grid.fit(X_train, y_train)
    # {'batch_size': 5,
    # 'drop_out': True,
    # 'epochs': 100,
    # 'optimizer': 'adagrad',
    # 'units': 50}
    # best score = 0.79


    nn = create_neural_network(optimizer='adagrad',drop_out=True,units=50)
    fit_records = nn.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=5)
    print 'evaluating neural network'
    score = nn.evaluate(X_test, y_test, batch_size=5)
    print '\nLoss: {}, Accuracy: {}' .format(score[0], score[1])

    y_test_predict = nn.predict_proba(X_testing)
    y_test_predict = y_test_predict.reshape(len(X_testing),1)
    save_submission(IDs, y_test_predict)
