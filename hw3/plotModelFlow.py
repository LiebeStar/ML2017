#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 01:02:29 2017

@author: liebe
"""

import keras
from keras.utils import plot_model

model = keras.models.load_model('model.h5')
plot_model(model, to_file='Structure.png')