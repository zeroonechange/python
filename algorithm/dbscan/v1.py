#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
""" 
Created on Thu Aug  3 21:49:21 2017 
 
@author: jack 
"""  
import numpy as np  
from numpy import *  
from matplotlib.pyplot import *    
import matplotlib.pyplot as plt  
from collections import defaultdict    
import random    
from math import radians, cos, sin, asin, sqrt   
   
#function to calculate distance   
# 欧拉距离   
def eclu(p1, p2):    
 return ((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)**(0.5)    
  
# 地球经纬度距离  
def dist(p1, p2):    
 return haversine(p1[0], p1[1], p2[0], p2[1])    
  
def haversine(lat1, lon1, lat2, lon2):   
    """  
    Calculate the great circle distance between two points   
    on the earth (specified in decimal degrees)  
    """    
    # 将十进制度数转化为弧度    
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])    
    
    # haversine公式    
    dlon = lon2 - lon1     
    dlat = lat2 - lat1     
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2    
    c = 2 * asin(sqrt(a))     
    r = 6371 # 地球平均半径，单位为公里    
    return c * r * 1000    
''''' 
深圳博物馆  
上梅林  22.5740523483,114.0660655322 
2763.99 ~ 2764 约 2.7 公里  = 2700米  
'''  
#print('distance:',  haversine(22.5492899209,114.0684141072,22.5740523483,114.0660655322 ) )  
  
def dbscan(radius, M, all_points):  
    #take radius and min. points = M    
    E = radius  
    minPts = M  
       
    #find out the core points    
    other_points =[]    
    core_points=[]        
    plotted_points=[]     
    for point in all_points:              
     point.append(0) # assign initial level 0    
     total = 0    
     for otherPoint in all_points:    
       distance = dist(otherPoint,point)    
       if distance<=E:    
         total+=1    
       
     if total > minPts:  
         print('add to core:', point)  
         core_points.append(point)    # add to core   
         plotted_points.append(point)    
     else:    
       other_points.append(point)    
    '''''  
    此轮下来 core_points = X(c)              包含了核心点 
            plotted_points = X(c)          也包含了核心点 
            other_points = X(bd) + X(noi)  包含了 边界点和噪声点   
    '''  
    print('total core point: ', len(core_points))
    #find border points    
    border_points=[]    
    for core in core_points:    
     for other in other_points:  
         d = dist(core, other)  
         if d<=E:     
             print('add to border---> dist:' , d)  
             border_points.append(other)    
             plotted_points.append(other)     
    ''''' 
    遍历 每个核心点  
        对其他点寻找  半径小于邻域的点   就是边界点   
        这样的话 border_points = X(bd)    边界点  
        plotted_points = X(c) + X(bd)  包含了核心点和边界点  
        这样子算 感觉好傻逼  在第一步计算核心点的时候一定可以做些什么  牺牲存储减少计算次数 
    '''  
          
    #implement the algorithm    
    cluster_label=0    
    for point in core_points:  # 遍历核心点 X(c)  
     if point[2]==0:    
       cluster_label+=1    
       print('add label to core point', cluster_label)  
       point[2]=cluster_label   # 给数据点增加一个维度 用来存放类的 label 信息  
       
     for point2 in plotted_points:   # 遍历核心点和边界点 X(c) + X(bd)   
       distance = dist(point2,point)  # 计算和核心点的距离  
       if point2[2] ==0 and distance<=E:   # 如果该点没有被其他类赋予label信息 并且在核心点的半径内   
         #print(point, point2)    
         point2[2] =point[2]             # 给该点指定label值  先来先赋值    
    ''''' 
    这一步做的是给核心点指定一个label   然后遍历 给边界点指定一个label  
    这里的计算可以通过上面寻找边界点省去  给边界点增加多维 再具体计算 
    plotted_points = X(c) + X(bd) 全部带有label  
    '''  
   
    #after the points are asssigned correnponding labels, we group them    
    cluster_list = defaultdict(lambda: [ [],[] ])    
    for point in plotted_points:    
     cluster_list[ point[2] ][0].append(point[1])  # 字典根据label统计核心点和边界点的具体数值   
     cluster_list[ point[2] ][1].append(point[0])    
      
    markers = ['+','*','.','d','^','v','>','<','p']  
       
    #plotting the clusters    
    i=0    
    print('the length of cluster:', len(cluster_list))  # 打印类的个数  
    for value in cluster_list:    
     cluster= cluster_list[value]    
     print('x:', cluster[0], 'y:', cluster[1])
     #saveFile( str(value) + '(3)_' + str(len(cluster[0])) + '.txt', cluster[0], cluster[1])
     plot(cluster[0], cluster[1], markers[i])   # x  y  marker  画出核心点和边界点  
     i = i +1  
     if i == 8:  
         i = 0  
       
    #plot the noise points as well    
    noise_points=[]    
    for point in all_points:    
     if not point in core_points and not point in border_points:    
         print('add to noise point', point)  
         noise_points.append(point)    
      
    noisex=[]    
    noisey=[]    
    for point in noise_points:    
     noisex.append(point[1])    
     noisey.append(point[0])    
    plot(noisex, noisey, "x")    # 画出噪声点  
    title(str(len(cluster_list))+" clusters created with E:"+str(E)+" Min Points:"+str(minPts)+" total points:"+str(len(all_points))+" noise Points:"+ str(len(noise_points)))    
    axis((80,130,18,50))    
    show()  
  
# 1. 随机生成100条记录来测试    
''''' 
all_points=[]   
for i in range(100):   
 randCoord = [random.randint(1,50), random.randint(1,50)]   
 if not randCoord in all_points: 
     print(randCoord)  #每个点在1 ~ 50 之间  
     all_points.append(randCoord)   
dbscan(10, 8, all_points) 
'''  

def saveFile(filename, line1, line2):
    file=open(filename,'w')
    for index, val in enumerate(line1):
        file.writelines( str(line1[index]) + '\t' +  str(line2[index]) + '\n')
    file.close()

    
def loadDataSet(fileName):  
    dataMat = []  
    fr = open(fileName)  
    for line in fr.readlines():  
        curLine = line.strip().split('\t')  
        #if float(curLine[1]) < 50:   # 过滤掉 维度为-1.00000的  
        #    continue  
        fltLine = [float(curLine[1]), float(curLine[0])] # 将字符串通过map转换成float类型  
        #fltLine = list(map(float, curLine))  # 将字符串通过map转换成float类型  
        dataMat.append(fltLine)  
    return dataMat  
'''
dataMat = mat(loadDataSet('/Users/jack/Desktop/dbscan/1000.txt')) 
plt.scatter( np.array(dataMat[:,1]), np.array(dataMat[:,0]) , c='g' ) 
plt.show() 
'''
# 半径以米为单位   个数为100家  
#dbscan(500000, 10, loadDataSet('111.txt'))
#dbscan(100000, 10, loadDataSet('3_2076.txt'))
#dbscan(50000, 10, loadDataSet('2(2)_427.txt'))
dbscan(5000, 2, loadDataSet('/Users/jack/Desktop/dbscan/1000.txt'))

