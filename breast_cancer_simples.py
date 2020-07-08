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

#compilar
classificador.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['binary_accuracy'])
#treino
classificador.fit(x=x_train,y=y_train,batch_size=10,epochs=100)     
# o output accuracy diz respeito a o teste aplicado na propria base de treino. desconsiderar

#teste 
predicted_data=classificador.predict(x=x_test)
# o retorno desta função é  à probabilidade prevista para cada um dos registros, segundo o modelo
# precisa-se comparar com os valores esperados no espaço amostral de teste (y_test). 
#Para isso é necessario normalizar as probabiliaddes para as determinadas classes. Segue abaixo
predicted_data= predicted_data>0.5

# ========= Avaliação simples de performance do modelo =========
#Com sklearn
from sklearn.metrics import confusion_matrix, accuracy_score

accuracy=accuracy_score(y_true=y_test,y_pred=predicted_data)
confusion_matrix=confusion_matrix(y_true=y_test,y_pred=predicted_data)
#Com Keras
evaluate=classificador.evaluate(x=x_test,y=y_test)

