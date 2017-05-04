#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:53:40 2017

@author: liebe
"""

import csv
import sys
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras import backend as K

img_rows = 48
img_cols = 48

def LoadTrainData(file_name):
    f = open (file_name, 'r')
    data = []
    for row in csv.reader(f):
        data.append(row)
    f.close()
    data = np.array(data)
    
    trainY = data[1:, 0]
    trainX = data[1:, 1]
    
    images = []
    for row in trainX:
        images.append(np.fromstring(row, dtype=int, sep=' '))
    trainX = np.array(images)
    
    return np.array(trainX, dtype=np.float), np.array(trainY, dtype=np.int)


"Keras 1.x.x image is saved as (channels, img_row, img_cols)"
"Keras 2.x.x image is saved as (img_row, img_cols, channels)"
def FormatData(trainX):
    if K.image_data_format() == 'channels_first':
        trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    trainX /= 255
    
    return trainX, input_shape



trainX, trainY = LoadTrainData(sys.argv[0])
trainX, input_shape = FormatData(trainX)

samples = int(trainX.shape[0] * 0.85)
x_val = trainX[samples:, :, :, :]
y_val = trainY[samples:]

# convert class vectors to binary class matrices
y_val = keras.utils.to_categorical(y_val, 7)


#%%
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
model = keras.models.load_model('model.h5')
model.summary()
#plot_model(model, to_file='model.png')

predict_result = model.predict_classes(x_val, batch_size=200)
conf_mat = confusion_matrix(np.argmax(y_val, axis=1),predict_result)
print(conf_mat)
plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.savefig('confusion matrix.png')
plt.show()

