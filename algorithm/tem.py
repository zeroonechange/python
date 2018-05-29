#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 23:29:38 2018

@author: jack
"""
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
x = np.arange(-5.0, 5.0, 0.01)
y1 = np.sin(x)
plt.figure(1)
plt.subplot(y1)