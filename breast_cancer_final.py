import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

X= pd.read_csv('data/entradas_breast.csv')
Y= pd.read_csv('data/saidas_breast.csv')

classificador = Sequential()
classificador.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'normal', input_dim = 30))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
classificador.fit(x=X, y=Y, batch_size = 10, epochs = 100)

classificador_json= classificador.to_json()

with open('breast_cancer_sequential_dense_nn','w') as json_file: 
    json_file.write(classificador_json)

classificador.save_weights('classificador_breast.h5')