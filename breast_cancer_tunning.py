# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:10:07 2020

@author: gabri
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV


X= pd.read_csv('data/entradas_breast.csv')
Y= pd.read_csv('data/saidas_breast.csv')


def criar_rede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador=Sequential()
    classificador.add(Dense(units= neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
    #Adicionando DropOut
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units= neurons, activation=activation, kernel_initializer=kernel_initializer))
    #Adicionando DropOut
    classificador.add(Dropout(0.2)) 
    classificador.add(Dense(units=1, activation='sigmoid'))
    classificador.compile(optimizer=optimizer, loss=loss,
                      metrics=['binary_accuracy'])
    return classificador

classificador= KerasClassifier(build_fn=criar_rede)
parametros={'batch_size':[10,30],
            'epochs':[50,100],
            'optimizer':['adam']}
