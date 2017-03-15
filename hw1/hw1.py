#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:35:15 2017

@author: liebe
"""

import random
import csv
import pandas as pd
import numpy as np

#data = pd.read_csv('train.csv', encoding='big5')

data = []

f = open('train.csv', 'r', encoding='big5')
for row in csv.reader(f):
    data.append(row)
f.close()


x_data = []
y_data = []
# 18 different data in one day 
for i in range(1, len(data), 18):
    # 24 hours split into two set
    tmp = []
    for j in range(i+9, i+10):
        for k in range(3, 12):
            if(data[j][k]=='NR'):
                data[j][k]=0
            tmp.append(float(data[j][k]))
    x_data.append(tmp);
    x_data[len(x_data)-1] = np.array(x_data[len(x_data)-1])
                 
for i in range(1, len(data), 18):
    tmp = []
    for j in range(i+9, i+10):
        for k in range(17, 26):
            if(data[j][k]=='NR'):
                data[j][k]=0
            tmp.append(float(data[j][k]))
    x_data.append(tmp);
    x_data[len(x_data)-1] = np.array(x_data[len(x_data)-1])
    
for i in range(1, len(data), 18):
    y_data.append(float(data[i+9][12]));

for i in range(1, len(data), 18):
    y_data.append(float(data[i+9][26]))
    

# ydata = b + w * xdata

init_b = random.uniform(-1000.0, 1000.0)
init_w = np.random.rand(1, 9)
lr = 1.0  #learning rate
lamda = random.uniform(0.001, 0.01) # for regularization
iteration = 120000;

b = init_b
w = init_w
b_lr = 1.0
w_lr = 1.0

#b_history = [b]
#w_history = [w]

for i in range(iteration):
    b_grad = 0.0
    w_grad = np.zeros((9, ), dtype= np.float)
    for n in range(len(x_data)):
        b_grad = b_grad - float(2.0*(y_data[n] - (b + np.dot(w, x_data[n]))))
        w_grad = w_grad - float(2.0*(y_data[n] - (b + np.dot(w, x_data[n])))) * x_data[n] + 2.0 * lamda * w 
        
    
    b_lr = b_lr + b_grad**2
    w_lr = w_lr + w_grad**2
    
    b = b - lr/np.sqrt(b_lr) * b_grad
    w = w - lr/np.sqrt(w_lr) * w_grad
                      
#    b_history.append(b)
#    w_history.append(w)
    
    print(str((i+1)*100/iteration) + '% ' + str(b_grad))
                      
###############  Testing Data  ##############

data = []
f = open('test_X.csv', 'r', encoding='big5')
for row in csv.reader(f):
    data.append(row)
f.close()

test_data = []
for i in range(0, len(data), 18):
    tmp = []
    for j in range(i+9, i+10):
        for k in range(2, len(data[j])):
            if(data[j][k]=='NR'):
                data[j][k]=0
            tmp.append(float(data[j][k]))
    test_data.append(tmp);
    test_data[len(test_data)-1] = np.array(test_data[len(test_data)-1])


##############  Running Model  ##############
predict = []
for i in range(len(test_data)):
    pm =  round( float(b + np.dot(w, test_data[i])) )
    if pm > 0:
        predict.append(pm)
    else:
        predict.append(0)

answer = [['id', 'value']]
for i in range(len(predict)):
    ID = 'id_' + str(i)
    tmp = [ID, predict[i]]
    answer.append(tmp)
    
    
f = open("test_Y.csv","w")
w = csv.writer(f)
w.writerows(answer)
f.close()

















    