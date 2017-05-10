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

num_classes = 7
img_rows = 48
img_cols = 48

def LoadTestData(file_name):
    f = open (file_name, 'r')
    data = []
    for row in csv.reader(f):
        data.append(row)
    f.close()
    data = np.array(data)

    testX = data[1:, 1]
    images = []
    for row in testX:
        images.append(row.split(' '))
    testX = np.array(images)
        
    return np.array(testX, dtype=np.float)

"Keras 1.x.x image is saved as (channels, img_row, img_cols)"
"Keras 2.x.x image is saved as (img_row, img_cols, channels)"
def FormatData(testX):
    if K.image_data_format() == 'channels_first':
        testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    testX /= 255
    
    return testX, input_shape

            
def WriteResult(result, file_name):
    f = open(file_name, 'w')
    f.write('id,label\n')
    for i in range(0, len(result)):
        f.write(str(i) + ',' + str(result[i]) + '\n')
    f.close()

        
# %%

if __name__ == "__main__":

    # data pre-processing
    testX = LoadTestData(sys.argv[1])
    testX, input_shape = FormatData(testX)
    x_test = testX    

    
# %% load model from file
    model = keras.models.load_model('model.h5')

# %% write file
    result = model.predict_classes(x_test, 128)
    WriteResult(result, sys.argv[2])










