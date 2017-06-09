# -*- coding: utf-8 -*-
"""
Created on Wed May  3 08:17:37 2017

@author: liebe
"""
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import sys
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.models import Sequential

class CFModel(Sequential):

    def __init__(self, n_users, m_items, k_factors, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Reshape((k_factors,)))
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Reshape((k_factors,)))
        super(CFModel, self).__init__(**kwargs)
        self.add(Merge([P, Q], mode='dot', dot_axes=1))

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


RATINGS_CSV_FILE = sys[1]+'train.csv'
TESTING_CSV_FILE = sys[1]+'test.csv'
RESULT_CSV_FILE = sys[2]
MODEL_WEIGHTS_FILE = 'CFModel.h5'
K_FACTORS = 120
RNG_SEED = 1446557

ratings = pd.read_csv(RATINGS_CSV_FILE, 
                      sep=',', 
                      encoding='latin-1', 
                      usecols=['UserID', 'MovieID', 'Rating'])
max_userid = ratings['UserID'].drop_duplicates().max()
max_movieid = ratings['MovieID'].drop_duplicates().max()
print( len(ratings), 'ratings loaded.')

         
shuffled_ratings = ratings.sample(frac=1., random_state=RNG_SEED)
Users = shuffled_ratings['UserID'].values
print('Users:', Users, ', shape =', Users.shape)
Movies = shuffled_ratings['MovieID'].values
print('Movies:', Movies, ', shape =', Movies.shape)
Ratings = shuffled_ratings['Rating'].values
print('Ratings:', Ratings, ', shape =', Ratings.shape)

#%%
model = CFModel(max_userid+1, max_movieid+1, K_FACTORS)
model.compile(loss='mse', optimizer='adamax')


callbacks = [EarlyStopping('val_loss', patience=2), 
             ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]
history = model.fit([Users, Movies], Ratings, batch_size=1000, epochs=30, validation_split=.1, verbose=1, callbacks=callbacks)

#%%
min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
print('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))

#%%
loss = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                     'training': [ math.sqrt(loss) for loss in history.history['loss'] ],
                     'validation': [ math.sqrt(loss) for loss in history.history['val_loss'] ]})
ax = loss.ix[:,:].plot(x='epoch', figsize={7,10}, grid=True)
ax.set_ylabel("root mean squared error")
ax.set_ylim([0.0,3.0]);
           
#%%
model.load_weights(MODEL_WEIGHTS_FILE)

tUsers = pd.read_csv(TESTING_CSV_FILE, sep=',', encoding='latin-1', usecols=['UserID'])
tMovies = pd.read_csv(TESTING_CSV_FILE, sep=',', encoding='latin-1', usecols=['MovieID'])


tUsers = np.array(tUsers)
tMovies = np.array(tMovies)
result = model.predict([tUsers, tMovies])
WriteResult(result, RESULT_CSV_FILE)


