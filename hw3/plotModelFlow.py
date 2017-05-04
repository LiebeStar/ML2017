#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 01:02:29 2017

@author: liebe
"""


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import sys
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

model = load_model('model.h5')
#%%
plot_model(model, to_file='model.png')