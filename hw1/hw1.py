#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:35:15 2017

@author: liebe
"""

import sys
import csv

f = open('train.csv', 'r', encoding="big5")
for row in csv.reader(f):
    print(row)
f.close()
