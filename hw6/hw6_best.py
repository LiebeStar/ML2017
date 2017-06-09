# -*- coding: utf-8 -*-
"""
Created on Wed May  3 08:17:37 2017

@author: liebe
"""
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.models import Sequential

class DeepModel(Sequential):

    def __init__(self, n_users, m_items, k_factors, p_dropout=0.3, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Reshape((k_factors,)))
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Reshape((k_factors,)))
        super(DeepModel, self).__init__(**kwargs)
        self.add(Merge([P, Q], mode='concat'))
        self.add(Dropout(p_dropout))
        self.add(Dense(k_factors, activation='relu'))
        self.add(Dropout(p_dropout))
        self.add(Dense(1, activation='linear'))

    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]

def WriteResult(result, file_name):
    f = open(file_name, 'w')
    f.write('TestDataID,Rating\n')
    for i in range(0, len(result)):
        
        if(result[i][0]>5.0):
            result[i] = 5.0
        
        if(result[i][0]<0):
            result[i] = 0
        
        f.write(str(i+1) + ',' + str(result[i][0]) + '\n')
    f.close()


RATINGS_CSV_FILE = sys[1] + 'train.csv'
TESTING_CSV_FILE = sys[1] + 'test.csv'
RESULT_CSV_FILE = sys[2]
MODEL_WEIGHTS_FILE = 'DNN.h5'
K_FACTORS = 120


#%%
model = DeepModel(6041, 3953, K_FACTORS)
model.compile(loss='mse', optimizer='adam')
           
#%%
model.load_weights(MODEL_WEIGHTS_FILE)

tUsers = pd.read_csv(TESTING_CSV_FILE, sep=',', encoding='latin-1', usecols=['UserID'])
tMovies = pd.read_csv(TESTING_CSV_FILE, sep=',', encoding='latin-1', usecols=['MovieID'])


tUsers = np.array(tUsers)
tMovies = np.array(tMovies)
result = model.predict([tUsers, tMovies])
WriteResult(result, RESULT_CSV_FILE)


