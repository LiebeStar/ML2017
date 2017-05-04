# -*- coding: utf-8 -*-
"""
Created on Wed May  3 08:17:37 2017

@author: liebe
"""
# %%

import sys
import csv
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator


#%%
num_classes = 7
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
    
    #Normalize
    trainX /= 255
    
    return trainX, input_shape
            

def cnn_dnn_model(input_shape):
    
    model = Sequential()
    model.add(Conv2D(20, kernel_size=(3, 3), input_shape=input_shape))
    model.add(keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(50, kernel_size=(3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(100, kernel_size=(3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    
    model.add(Conv2D(200, kernel_size=(3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Flatten())
    
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(num_classes, activation='softmax'))

    return model
# %%

if __name__ == "__main__":

    # data pre-processing
    trainX, trainY = LoadTrainData(sys.argv[0])

    trainX, input_shape = FormatData(trainX)
    
    samples = int(trainX.shape[0] * 0.85)
    x_train = trainX[:samples, :, :, :]
    y_train = trainY[:samples]
    x_vali = trainX[samples:, :, :, :]
    y_vali = trainY[samples:]
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_vali = keras.utils.to_categorical(y_vali, num_classes)


# %% construct model
    model = cnn_dnn_model(input_shape)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(),
                  #optimizer=keras.optimizers.Adam(),
                  # optimizer=keras.optimizers.SGD(lr=0.01, decay=0.0),
                  metrics=['accuracy'])
    model.summary()
    
# %% fit model
    # fit with image generator
    dataGen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True)
    
    dataGen.fit(x_train)
    
    epochs = 800
    batch_size = 128
    history = model.fit_generator(dataGen.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=len(x_train)/32,
                                  epochs=epochs,
                                  validation_data=(x_vali, y_vali))
    """
    # fit without image generator
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_vali, y_vali),
              callbacks=[history, tbCallBack])
    """
 #%%           
    model.save('model.h5')
    
    score = model.evaluate(x_vali, y_vali, verbose=0)
    print('Total loss:', score[0])
    print('Accuracy:', score[1])










