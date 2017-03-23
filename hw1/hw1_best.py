#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:35:15 2017

@author: liebe
"""

import random
import csv
import copy
import sys
import numpy as np

data = []

f = open( sys.argv[1], 'r', encoding='big5')
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
    x_data.append(tmp)
    x_data[len(x_data)-1] = np.array(x_data[len(x_data)-1])

for i in range(1, len(data), 18):
    tmp = []
    for j in range(i+9, i+10):
        for k in range(17, 26):
            if(data[j][k]=='NR'):
                data[j][k]=0
            tmp.append(float(data[j][k]))
    x_data.append(tmp)
    x_data[len(x_data)-1] = np.array(x_data[len(x_data)-1])
    
for i in range(1, len(data), 18):
    y_data.append(float(data[i+9][12]))

for i in range(1, len(data), 18):
    y_data.append(float(data[i+9][26]))

# ydata = b + w * xdata

#init_b = random.uniform(-1000.0, 1000.0)
#init_w = np.random.rand(1, 9)
#lamda = random.uniform(0.001, 0.01) # for regularization

lr = 1.0  #learning rate

init_b = 143.9310038399983
init_w = np.array( [
7.958930141278124371e-01,
3.416067742381843075e-01,
6.186365119737545770e-01,
5.372440512447886896e-01,
3.346187037680361520e-01,
2.418090409083799575e-01,
4.397033172869396767e-01,
4.662250012569726376e-01,
5.030662612860843375e-01 ] ) 
lamda = 0.008570385672974277                 
iteration = 100000;

b = copy.copy(init_b)
w = copy.copy(init_w)
b_lr = 1.0
w_lr = 1.0
now = 1

for i in range(iteration):
    b_grad = 0.0
    w_grad = np.zeros((9, ), dtype= np.float)
    for n in range(len(x_data)):
        b_grad = b_grad - float(2.0*(y_data[n] - (b + np.dot(w, x_data[n]))))
        w_grad = w_grad - float(2.0*(y_data[n] - (b + np.dot(w, x_data[n])))) * x_data[n] + 2.0*lamda*w        
    
    b_lr = b_lr + b_grad**2
    w_lr = w_lr + w_grad**2
    
    b = b - lr/np.sqrt(b_lr) * b_grad
    w = w - lr/np.sqrt(w_lr) * w_grad  
    
    if((i+1)*100/iteration >= now):
        print(str(now) + '%')
        now = now + 1
                      
###############  Testing Data  ##############

data = []
f = open(sys.argv[2], 'r', encoding='big5')
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
    pm =  round( float( b + np.dot(w, test_data[i])) - 0.5)
    if pm > 0:
        predict.append(pm)
    else:
        predict.append(0)

answer = [['id', 'value']]
for i in range(len(predict)):
    ID = 'id_' + str(i)
    tmp = [ID, predict[i]]
    answer.append(tmp)
    
    
f = open(sys.argv[3],"w")
w = csv.writer(f)
w.writerows(answer)
f.close()

















    