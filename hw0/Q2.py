#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 02:28:39 2017

@author: liebe
"""

import sys
import numpy as np
from PIL import Image

img1 = np.array( Image.open(sys.argv[1]) )
img2 = np.array( Image.open(sys.argv[2]) )

for i in range( 0, len(img1)):
    for j in range( 0, len(img1[i])):
        if np.array_equal(img1[i][j], img2[i][j]):
            img2[i][j] = 0
                
result = Image.fromarray(img2)
result.save('ans_two.png')
