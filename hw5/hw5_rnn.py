#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:27:00 2017

@author: liebe
"""

import numpy as np
import sys
from keras import backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam

import pickle

test_path = sys.argv[1]
output_path = sys.argv[2]

#####################
###   parameter   ###
#####################
split_ratio = 0.001
embedding_dim = 100
nb_epoch = 1000
batch_size = 64


################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r', encoding='utf-8', errors='ignore') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

#########################
###   Main function   ###
#########################

#%%


### read training and testing data
#(Y_data,X_data,tag_list) = read_data(train_path,True)
(_, X_test,_) = read_data(test_path,False)
all_corpus = X_test
print ('Find %d articles.' %(len(all_corpus)))


tag_list = ['SCIENCE-FICTION',
 'SPECULATIVE-FICTION',
 'FICTION',
 'NOVEL',
 'FANTASY',
 "CHILDREN'S-LITERATURE",
 'HUMOUR',
 'SATIRE',
 'HISTORICAL-FICTION',
 'HISTORY',
 'MYSTERY',
 'SUSPENSE',
 'ADVENTURE-NOVEL',
 'SPY-FICTION',
 'AUTOBIOGRAPHY',
 'HORROR',
 'THRILLER',
 'ROMANCE-NOVEL',
 'COMEDY',
 'NOVELLA',
 'WAR-NOVEL',
 'DYSTOPIA',
 'COMIC-NOVEL',
 'DETECTIVE-FICTION',
 'HISTORICAL-NOVEL',
 'BIOGRAPHY',
 'MEMOIR',
 'NON-FICTION',
 'CRIME-FICTION',
 'AUTOBIOGRAPHICAL-NOVEL',
 'ALTERNATE-HISTORY',
 'TECHNO-THRILLER',
 'UTOPIAN-AND-DYSTOPIAN-FICTION',
 'YOUNG-ADULT-LITERATURE',
 'SHORT-STORY',
 'GOTHIC-FICTION',
 'APOCALYPTIC-AND-POST-APOCALYPTIC-FICTION',
 'HIGH-FANTASY']


### tokenizer for all data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_corpus)

with open('word_index.pkl', 'rb') as pickle_file:
    word_index = pickle.load(pickle_file)
    tokenizer.word_index = word_index

### convert word sequences to index sequence
print ('Convert to index sequences.')
#train_sequences = tokenizer.texts_to_sequences(X_data)
test_sequences = tokenizer.texts_to_sequences(X_test)

### padding to equal length
print ('Padding sequences.')
#train_sequences = pad_sequences(train_sequences)
max_article_length = 306
test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)


### build model
print ('Building model.')
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                    embedding_dim,
                    input_length=max_article_length,
                    trainable=False))

model.add(GRU(128,activation='tanh',dropout=0.25))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(80,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(38,activation='sigmoid'))

adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=[f1_score])


#%%
model.load_weights('best.hdf5')
Y_pred = model.predict(test_sequences)
thresh = 0.4
with open(output_path,'w') as output:
    print ('\"id\",\"tags\"',file=output)
    Y_pred_thresh = (Y_pred > thresh).astype('int')
    for index,labels in enumerate(Y_pred_thresh):
        labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
        labels_original = ' '.join(labels)
        print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
    
