# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:36:46 2017

@author: jack
"""
import numpy as np
from collections import defaultdict    

def saveFile(filename, line1, line2):
    file=open(filename,'w')
    for index, val in enumerate(line1):
        file.writelines(line1[index] + '\t' +  line2[index] + '\n')
    file.close()
    
cluster = []
for i in range(100):
    cluster.append('dasdds')

print(cluster)
value =1
saveFile( str(value) +'_'+ str(len(cluster[0])) + '.txt', cluster,cluster)
