#!/usr/bin/env python
# -- coding: utf-8 --
#%%
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import pandas
import numpy as np
#%%
x_val=[]
y_val=[]
train_data = pandas.read_csv('train.csv') #training data

for i in range(0,4000):
    temp = np.array(list(map(int, train_data.loc[i, 'feature'].split())))
    temp = temp.reshape(1, 48,48, 1)
    x_val.append(temp)
    temp = np.zeros((7, ),dtype=np.int)
    temp[int(train_data.loc[i, 'label'])] = 1
    y_val.append(temp)
x_val = np.array(x_val)
y_val = np.array(y_val)

#%%

emotion_classifier = load_model('model.h5')
layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[0:])

input_img = emotion_classifier.input
name_ls = ["conv2d_24"]
collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

choose_id = 17
photo = x_val[choose_id]
for cnt, fn in enumerate(collect_layers):
    im = fn([photo, 0]) #get the output of that layer
    fig = plt.figure(figsize=(15, 13))
    nb_filter = im[0].shape[3]
    for i in range(nb_filter):
        ax = fig.add_subplot(nb_filter/15, 15, i+1)
        ax.imshow(im[0][0, :, :, i], cmap='BuGn')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
    