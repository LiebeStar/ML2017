#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:56:12 2017

@author: liebe
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

path = '/Users/liebe/Liebe/NTU/Second Semester/ML/Homework/hw4/faceExpressionDatabase/'
person = 'ABCDEFGHIJ'

oriImg = []
for i in range(0,10):
    for j in range(0,10):
        im1 = mpimg.imread( path + person[i] + '0' + str(j) + '.bmp')
        oriImg.append(im1.flatten())
oriImg = np.asarray(oriImg).astype(float)


#%%
aveImg = oriImg.mean(axis=0, keepdims=True)
cmap = plt.get_cmap('gray')
plt.imshow(aveImg.reshape((64,64)), cmap)


#%%
subImg = oriImg - aveImg
U, s, V = np.linalg.svd(subImg, full_matrices=False)
S = np.diag(s)


#%%
# Q1-1: Print top 9 eigenfaces
f, eigen = plt.subplots(3, 3)
for i in range(0,3):
    for j in range(0, 3):
        eigen[i,j].imshow( V[i*3+j].reshape((64,64)) , cmap)
        eigen[i,j].set_xlabel("#" + str(i*3+j) + " eigenface")
        eigen[i,j].set_xticks([])
        eigen[i,j].set_yticks([])

#%%
# Q1-2: Print origin image and recovered image, recovered by top 5 eigenfaces

recoverImg = aveImg
for i in range(0,5):
    weight = np.dot( subImg, V[i] )
    recoverImg = recoverImg + np.dot( weight.reshape((100,1)), V[i].reshape((1, 4096)))

f, ori = plt.subplots(10, 10)
for i in range(0, 10):
    for j in range(0, 10):
        ori[i,j].imshow( oriImg[i*10+j].reshape((64,64)) , cmap)
        ori[i,j].set_xticks([])
        ori[i,j].set_yticks([])

f, rec = plt.subplots(10, 10)
for i in range(0, 10):
    for j in range(0, 10):
        rec[i,j].imshow( recoverImg[i*10+j].reshape((64,64)) , cmap)
        rec[i,j].set_xticks([])
        rec[i,j].set_yticks([])

#%%
# Q1-3: 

for k in range(1, 101):
    recoverImg = aveImg
    for i in range(0,k):
        weight = np.dot( subImg, V[i] )
        recoverImg = recoverImg + np.dot( weight.reshape((100,1)), V[i].reshape((1, 4096)))
    
    tmp = oriImg - recoverImg
    RMSE = ((np.sum(tmp**2)/(4096*100))/65535)**0.5
    if(RMSE<=0.01):
        break
    
    
recoverImg = aveImg
for i in range(0,k):
    weight = np.dot( subImg, V[i] )
    recoverImg = recoverImg + np.dot( weight.reshape((100,1)), V[i].reshape((1, 4096)))
    
    
f, ori = plt.subplots(10, 10)
for i in range(0, 10):
    for j in range(0, 10):
        ori[i,j].imshow( oriImg[i*10+j].reshape((64,64)) , cmap)
        ori[i,j].set_xticks([])
        ori[i,j].set_yticks([])





