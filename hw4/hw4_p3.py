# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:51:44 2017

@author: liebe
"""

import sys
import numpy as np
from sklearn.cluster import KMeans

train = np.load(sys.argv[1])

x = []
for i in range(0, 200):
    tmp = np.reshape(train[str(i)], train[str(i)].shape[0] * train[str(i)].shape[1])
    x.append( tmp.var() )

x = np.reshape(x, (200,1))
kmeans = KMeans(n_clusters=60, random_state=42).fit(x)

y = np.sort(kmeans.cluster_centers_.T).T

f = open(sys.argv[2], 'w')
f.write('SetId,LogDim\n')
for i in range(0, 200):
    for j in range(0, 59):
        if((x[i][0] >= y[j][0]) and (x[i][0] <= y[j+1][0])):
            if(abs(x[i][0]-y[j][0]) < abs(x[i][0]-y[j+1][0])):
                f.write(str(i) + ',' + str( np.log(j+1.0) ) + '\n')
            else:
                f.write(str(i) + ',' + str( np.log(j+2.0) ) + '\n')
            break
        elif(j==58 and x[i][0]>=y[59][0]):
            f.write(str(i) + ',' + str( np.log(j+2.0) ) + '\n')
        elif(j==0 and x[i][0]<=y[0][0]):
            f.write(str(i) + ',' + str( np.log(j+1.0) ) + '\n')
f.close()
