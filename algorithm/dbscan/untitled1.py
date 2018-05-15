#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 23:03:43 2018

@author: jack
"""
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

data = pd.read_csv('/Users/jack/Desktop/dbscan/examples/spx.csv', index_col =0, parse_dates = True)
spx = data['SPX']
spx.plot(ax=ax, style='k-')

crisis_data = [
    (datetime(2007, 10, 11), 'Peak of bull market'),
    (datetime(2008, 3, 12), 'Bear Stearns Fails'),
    (datetime(2008, 9, 15), 'Lehman Bankruptcy')
]

for date, label in crisis_data:
    ax.annotate(label, xy=(data, spx.asof(date) + 75), 
               xytext=(date, spx.asof(date) + 225),
               arrowprops=dict(facecolor='black', headwidth=4, width=2, headlength=4),
               horizontalalignment='left', verticalalignment='top')
    
# Zoom in on 2007-2010
ax.set_xlim('1/1/2007', '1/1/2011')
ax.set_ylim([600, 1800])
ax.set_title('Important dates in the 2008-2009 financial crisis')