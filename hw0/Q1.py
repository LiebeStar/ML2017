# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def isanumber(a):

    bool_a = True
    try:
        bool_a = float(repr(a))
    except:
        bool_a = False

    return bool_a


import sys
import numpy as np

matrixA = np.loadtxt( sys.argv[1], dtype='i', delimiter=',')
matrixB = np.loadtxt( sys.argv[2], dtype='i', delimiter=',')

mul = np.dot(matrixA, matrixB)

result = []
for i in range( 0, len(mul)):
    if isanumber(mul[i]):
        result.append(mul[i])
    else:
        for j in range( 0, len(mul[i])):
            result.append(mul[i][j])
   
result = sorted(result)

with open('ans_one.txt', 'w') as f:
    for e in range(0, len(result)):
        f.write(str(result[e])+'\n')