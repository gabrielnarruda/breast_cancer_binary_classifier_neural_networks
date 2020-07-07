# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 20:45:07 2020

@author: gabrielnarruda
"""

import pandas as pd

X= pd.read_csv('data/entradas_breast.csv')
Y= pd.read_csv('data/saidas_breast.csv')
 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense

classificador=Sequential()
# Camada de Entrada
classificador.add(Dense(units= 16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
 # input_dim = quantidade de atributos de X
#camada de saida
classificador.add(Dense(units=1, activation='sigmoid'))
# activation='sigmoid' por ser um problema de classificação


print(classificador)