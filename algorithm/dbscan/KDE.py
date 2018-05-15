#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import math

np.random.seed(1)    # 系统生成随机值的整数值 如果指定则生成随机数相同   
N = 20
M = np.concatenate((np.random.normal(0, 1, int(0.3 * N)), np.random.normal(5, 1, int(0.7 * N))))
X = M[:, np.newaxis]    #将一维转化成数组 

X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
bins = np.linspace(-5, 10, 10)

# 一张图放多张子图 这里是2X2=4张 share属性是参考系默认一样
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
# 返回图形对象 和 坐标系的对象数组
fig.subplots_adjust(hspace=0.05, wspace=0.05)  #调整间距

# 直方图 1 'Histogram=直方图'   
ax[0, 0].hist(X[:, 0], bins=bins, fc='#AAAAFF', normed=True)
ax[0, 0].text(-3.5, 0.31, 'Histogram')  # 按坐标轴原点来的

# 直方图 2 'Histogram, bins shifted'   
ax[0, 1].hist(X[:, 0], bins=bins + 0.75, fc='#AAAAFF', normed=True)
ax[0, 1].text(-3.5, 0.31, 'Histogram, bins shifted')

# 核密度估计 1 'tophat KDE'
kde = KernelDensity(kernel='tophat', bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot)

ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax[1, 0].text(-3.5, 0.31, 'Tophat Kernel Density')

# 核密度估计 2 'Gaussian KDE'
kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
log_dens = kde.score_samples(X_plot) #点x对应概率的log值，要使用exp求指数还原
print(len(log_dens))
for x in log_dens:
    print('the probablity is:' + str(math.exp(x)))
    
ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
ax[1, 1].text(-3.5, 0.31, 'Gaussian Kernel Density')

for axi in ax.ravel():
    axi.plot(X[:, 0], np.zeros(X.shape[0])-0.01, '+k')
    axi.set_xlim(-4, 9)
    axi.set_ylim(-0.02, 0.34)
for axi in ax[:, 0]:
    axi.set_ylabel('Normalized Density')
for axi in ax[1, :]:
    axi.set_xlabel('x')
plt.show()