#!/usr/bin/env python
# -- coding: utf-8 --

import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_ascent(num_step,input_image_data,iter_func):
    print('Start')    
    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iter_func([input_image_data, 0])
        input_image_data += grads_value * num_step

        print('Current loss value:', loss_value)

    print('End')    
    return deprocess_image(input_image_data.reshape(48,48)), loss_value
#%%
    
emotion_classifier = load_model('model.h5')
layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[0:])
input_img = emotion_classifier.input

name_ls = ["conv2d_24"]
collect_layers = [ layer_dict[name].output for name in name_ls ]

nb_filter = 150
for cnt, c in enumerate(collect_layers):
    filter_imgs = []
    loss_vals = []
    for filter_idx in range(nb_filter):
        input_img_data = np.random.random((1, 48, 48, 1)) # random noise
        target = K.mean(c[:, :, :, filter_idx])
        grads = normalize(K.gradients(target, input_img)[0])
        iterate = K.function([input_img, K.learning_phase()], [target, grads])

        (img, loss_val) = grad_ascent(0.2, input_img_data, iterate)
        filter_imgs.append(img)
        loss_vals.append(loss_val)
#%%        
fig = plt.figure(figsize=(15, 18))
for i in range(nb_filter):
    ax = fig.add_subplot(nb_filter/15, 15, i+1)
    ax.imshow(filter_imgs[i], cmap='BuGn')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel('{:.3f}'.format(loss_vals[i]))
    plt.tight_layout()
    fig.suptitle('Filters of layer {}'.format(name_ls[cnt]))

