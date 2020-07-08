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

# ===== Construindo a rede
# === Instanciando o tipo 
classificador=Sequential()
# === Camada de Entrada
classificador.add(Dense(units= 16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
 # input_dim = quantidade de atributos de X. OBS: apenas para a primeira camada
 
 #=== Adicioanndo segunda camada
classificador.add(Dense(units= 16, activation='relu', kernel_initializer='random_uniform'))
# Testar a melhor combinação dos parâmetros, seja quantidade de neurônios, funções de ativação, kernel  

#=== Camada de saida
classificador.add(Dense(units=1, activation='sigmoid'))
# Escolher corretamente a função de ativação para a saída. OBS activation='sigmoid' por ser um problema de classificação


# == Modelando optimizer 

otimizador = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.001, beta_2=0.001, clipvalue=0.5)
# pesquisar sobre os parâmetros  betas e clipvalue na api

#==== Compilar
# o paramerto optimizer é o algoritmo que realiza o ajuste dos pesos
classificador.compile(optimizer=otimizador, loss='binary_crossentropy',
                      metrics=['binary_accuracy'])
#treino
classificador.fit(x=x_train,y=y_train,batch_size=10,epochs=100)     
# o output accuracy diz respeito a o teste aplicado na propria base de treino. desconsiderar

#=== Visualizando os pesos da rede
pesos_camada_0=classificador.layers[0].get_weights()
# = array[0] diz respeito aos pesos da camada de entrada à camada de saída.
# = array[1] diz respeiro aos pesos da Bias Unit
pesos_camada_1=classificador.layers[1].get_weights()
pesos_camada_2=classificador.layers[2].get_weights()


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

