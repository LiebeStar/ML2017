#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:13:09 2017

@author: liebe
"""

import word2vec
import numpy as np
import nltk


# DEFINE your parameters for training
MIN_COUNT = 20
WORDVEC_DIM = 300
WINDOW = 5
NEGATIVE_SAMPLES = 3
ITERATIONS = 200
MODEL = 1
LEARNING_RATE = 0.1

# train model
word2vec.word2vec(
    train = 'all.txt',
    output = 'model.bin',
    cbow = MODEL,
    size = WORDVEC_DIM,
    min_count = MIN_COUNT,
    window = WINDOW,
    negative = NEGATIVE_SAMPLES,
    iter_ = ITERATIONS,
    alpha = LEARNING_RATE,
    verbose = False)

# load model for plotting
model = word2vec.load('model.bin')

vocabs = []                 
vecs = []                   
for vocab in model.vocab:
    vocabs.append(vocab)
    vecs.append(model[vocab])
vecs = np.array(vecs)[:800]
vocabs = vocabs[:800]

'''
Dimensionality Reduction
'''
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(vecs)


'''
Plotting
'''
import matplotlib.pyplot as plt
from adjustText import adjust_text

# filtering
use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]


plt.figure(figsize = (16,12))
texts = []
for i, label in enumerate(vocabs):
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
            and all(c not in label for c in puncts)):
        x, y = reduced[i, :]
        texts.append(plt.text(x, y, label))
        plt.scatter(x, y)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

plt.savefig('hp.png', dpi=600)
plt.show()






