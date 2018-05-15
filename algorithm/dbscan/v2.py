#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 00:43:14 2017

@author: jack
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
# matplotlib inline
X1, y1=datasets.make_circles(n_samples=5000, factor=.6,noise=.05)
# n_samples样本数 n_features特征数量  centers类别数 cluster_std每个类别的方差
X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]],random_state=9)
X = np.concatenate((X1, X2))
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()

from sklearn.cluster import DBSCAN
y_pred = DBSCAN(eps=0.1, min_samples=10).fit_predict(X)

tp = []
for x in y_pred:
    if x not in tp:
        tp.append(x)
    if x==-1:
        print('I found u ')
print(tp)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()


'''
from sklearn.neighbors import kde
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
kde = kde.KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)

# [-0.41075698 -0.41075698 -0.41076071 -0.41075698 -0.41075698 -0.41076071]
print(kde.score_samples(X)) 

# [ 0.66314807  0.66314807  0.6631456   0.66314807  0.66314807  0.6631456 ]
print(np.exp(kde.score_samples(X)))
'''