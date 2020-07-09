# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 23:22:39 2020

@author: gabrielnarruda
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score


X= pd.read_csv('data/entradas_breast.csv')
Y= pd.read_csv('data/saidas_breast.csv')


def criar_rede():
    classificador=Sequential()
    classificador.add(Dense(units= 16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
    #Adicionando DropOut
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units= 16, activation='relu', kernel_initializer='random_uniform'))
    #Adicionando DropOut
    classificador.add(Dropout(0.2)) 
    classificador.add(Dense(units=1, activation='sigmoid'))
    otimizador = keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
    classificador.compile(optimizer=otimizador, loss='binary_crossentropy',
                      metrics=['binary_accuracy'])
    return classificador


#=== Criando classificador com interface sklearn OBS:Pesquisar mais
classificador= KerasClassifier(build_fn=criar_rede,
                               epochs=100,
                               batch_size=10)

#=== CrossValidation
resultados= cross_val_score(estimator=classificador, X=X,y=Y, cv=10, scoring='accuracy')
## parâmetros CV diz a qualtidade de divisões do espaço amostral
## O retorno da função é um array com todos os conjuntos de teste 

media_performance_modelo= resultados.mean()
std=resultados.std()