#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:35:15 2017

@author: liebe
"""

import sys
import csv
import numpy as np

def fread(fileName):
    data = []
    f = open( fileName, 'r', encoding='big5')
    for row in csv.reader(f):
        data.append(np.array(row))
    f.close()
    return data

def mean(data, size):
    Sum = np.zeros(size)
    for i in range(len(data)):
        Sum += data[i]
    return Sum/float(len(data))

def covariance_matrix(data, size, data_mean):
    Sum = np.zeros( (size, size))
    data = np.matrix(data)
    for i in range(len(data)):
        tmp = np.transpose(data[i])-data_mean
        Sum = Sum + tmp * np.transpose(tmp)
    return Sum/float(len(data))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))



data = fread('X_train')
y_data = fread('Y_train')

x_data = []
for i in range(len(y_data)):
    y_data[i] = float(y_data[i][0])
    tmp = []
    for j in range(len(data[i+1])):
        tmp.append(float(data[i+1][j]))
    x_data.append(np.array(tmp))
    

x_data1 = []
x_data2 = []
for i in range(len(y_data)):
    # <50K in x_data1
    if (y_data[i]==0):
        x_data1.append(x_data[i])
    else:
        x_data2.append(x_data[i])


mean1 = np.transpose( np.matrix( mean(x_data1,len(x_data1[0])) ) )
mean2 = np.transpose( np.matrix( mean(x_data2,len(x_data2[0])) ) )

all_size = float(len(x_data))
size1 = float(len(x_data1))
size2 = float(len(x_data2))

cm1 = covariance_matrix(x_data1, len(x_data1[0]), mean1)
cm2 = covariance_matrix(x_data2, len(x_data2[0]), mean2)
cm = (cm1*size1 + cm2*size2)/all_size
     
tcm = np.linalg.inv(cm)

w = np.transpose( np.transpose(mean1-mean2) * np.linalg.inv(cm) )
b = ( -0.5 * np.transpose(mean1) * np.linalg.inv(cm) * mean1 ) + ( 0.5 * np.transpose(mean2) * np.linalg.inv(cm) * mean2 ) + np.log(size1/size2)


data = fread('X_test')
test_X = []
for i in range(1, len(data)):
    tmp = []
    for j in range(len(data[i])):
        tmp.append(float(data[i][j]))
    test_X.append(np.array(tmp))


predict = []
for i in range(0, len(test_X)):
    z = np.matrix(test_X[i])*w + b
    if(float(sigmoid(z[0][0])) >= 0.55):
        predict.append(0)
    else:
        predict.append(1)

    #predict.append( 1-round( float(sigmoid(z[0][0]))) )


answer = [['id', 'label']]
for i in range(len(predict)):
    tmp = [i+1, predict[i]]
    answer.append(tmp)
    
    
f = open('test_Y.csv','w')
w = csv.writer(f)
w.writerows(answer)
f.close()



    