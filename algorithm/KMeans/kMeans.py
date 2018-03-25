# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 将字符串通过map转换成float类型
        dataMat.append(fltLine)
    return dataMat

# 计算欧拉距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# 构建一个包含k个随机质心的集合 
# 随机质心在整个数据集边界之内 通过找到每一维的最大/小 
# 生成0到1.0的随机数并通过取值范围和最小值确保随机点在数据边界之内
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

'''
创建k个点作为起始质心(一般随机)
当任意一个点的簇分配结果发生改变时
    对数据集中每个数据点
        对每个质心
            计算质心与数据之间的距离
        将数据点分配到最近的簇
    对每个簇，计算簇中所有点的均值并将均值作为质心达到收敛的效果
    
    k的取法？ 
    随机点的选择 达到最小的计算次数? 
centroids 为质心矩阵
clusterAssment 第一列是簇索引值  第二列是误差值
'''
def kMeans(dataSet, k, distMeas=distEclud, createCent = randCent):
    m = shape(dataSet)[0]  # 获取行数
    clusterAssment = mat(zeros((m, 2)))  #构建 m x 2的零矩阵  存下标和距离
    centroids = createCent(dataSet, k)  # 构建包含k个随机质心的集合
    clusterChanged = True
    while clusterChanged:   
        clusterChanged = False
        for i in range(m):  
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2 #直接赋值
        #print(centroids)
        #print(clusterAssment)
        for cent in range(k):
            #print( nonzero(clusterAssment[:,0].A == cent) )
            # clusterAssment的第一列[0]为cent 也就是k的索引   
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            #print('before mean: %s'%ptsInClust)  # 收集
            ave = mean(ptsInClust, axis=0)  # 对矩阵的列方向进行均值计算
            #print('ave: %s'%ave)
            centroids[cent, :] = ave
    return centroids, clusterAssment

 
'''
dataMat = mat(loadDataSet('testSet.txt'))
plt.scatter( np.array(dataMat[:,0]), np.array(dataMat[:,1]) , c='g' )
plt.show()
'''


'''
centroids = randCent(dataMat, 2)
print(min(dataMat[:,1]))
print(max(dataMat[:,1]))
print(centroids)
print(distEclud(centroids[0], centroids[1]))
'''

'''
myCentroids, clustAssing = kMeans(dataMat, 4)
plt.scatter( np.array(myCentroids[:,0]), np.array(myCentroids[:,1]), c='r'  )
plt.show()
'''



'''
SSE(Sum of Squared Error, 误差平方和)  对误差取了平方，重视了远离中心的点 
可能只是局部最优而不是全局最优  如何改进  
二分K-均值  先将所有点分为一个簇 然后再分为俩个簇 按照降低SSE的原则进行划分 一直满足指定簇数为止 

将所有点看出是一个簇
当簇数小于k时
对每一个簇
    计算总误差
    在给定的簇上面进行K-均值聚类(k=2)
    计算将该簇一分为二之后的总误差
选择使得误差最小的那个簇进行划分操作
'''
def bitKmeans(dataSet, k, distMeas = distEclud):
    m = shape(dataSet)[0]               # 得到行数
    clusterAssment = mat(zeros((m,2)))  # 构建m x 2 的矩阵  用来存储每个点的分配簇 以及平方误差
    centroid0 = mean(dataSet, axis=0).tolist()[0]  # 创建一个初始簇  全部数据的中心点 
    print('m:', m)              # 60
    print('k:', k)              # 3
    print(type(centroid0))          # list
    print(len(centroid0))           # 2
    print(centroid0)     #[-0.15772275000000002, 1.2253301166666664]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2   # 先为每个点确定欧氏距离
        #print('every point of eclud distance is:', clusterAssment[j,1])
    while (len(centList) < k): 
        lowerestSSE = inf    #无穷大
        for i in range(len(centList)):
            print(' for looper:', i)
            #  该簇中所有点看做是一个数据集  
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0], :] 
            # 将数据集将产生俩个质心 和 每个簇的误差值  第一次是60个 第二次是20个  第三次是40个
            print('划分前:\n', ptsInCurrCluster)
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            print('新的俩个质心:\n', centroidMat)
            print('新的俩个质心的下标和距离:\n', splitClustAss)  
            # 全部
            sseSplit = sum(splitClustAss[:,1])  
            # 筛选出 属于本簇的点误差值 
            print('其他点的误差的下标值', nonzero(clusterAssment[:,0].A!=i)[0])
            sseNoSplit = sum( clusterAssment[  nonzero(clusterAssment[:,0].A!=i)[0],  1] )
            print('划分前 当前总误差:', sseSplit, '\t 上次划分总误差', sseNoSplit)
            if (sseSplit + sseNoSplit) < lowerestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat          # 新的俩个质心
                bestClusAss = splitClustAss.copy()  # 要划分数据集的新的距离(针对俩个新的质心)
                lowerestSSE = sseSplit + sseNoSplit
                # 更新簇的分配结果  2阶划分后第一列只能是0或1对应俩个质心矩阵   新的放前面  1是旧的 
                # 前一个质心为原来的下标  后一个质心为新的下标  
                # 第一次为0  1  将坐标为0的改为0  将坐标为1的改为数组长度
                # 
                bestClusAss[ nonzero(bestClusAss[:,0].A==0)[0], 0] = bestCentToSplit
                bestClusAss[ nonzero(bestClusAss[:,0].A==1)[0], 0] = len(centList)
                print('原来的簇点:', bestCentToSplit, '\t 新加的簇点:', len(centList))                
                centList[bestCentToSplit] = bestNewCents[0,:]  # 先覆盖原来的质点
                centList.append(bestNewCents[1,:])             # 再添加新质点
                #  更新距离
                clusterAssment[ nonzero(clusterAssment[:,0].A== bestCentToSplit)[0], :] = bestClusAss
    return centList, clusterAssment

if __name__ == "__main__":
    dataMat3 = mat(loadDataSet('testSet2.txt'))
    plt.scatter( np.array(dataMat3[:,0]), np.array(dataMat3[:,1]), c='g')
    plt.show()
    
    cenList, myNewAssment = bitKmeans(dataMat3, 3)
    x = []
    y = []
    for ma in cenList:
        x.append(ma[0,0])
        y.append(ma[0.1])
    plt.scatter( x, y , c='r')
    plt.show()
    
    
    print(cenList)
    print(myNewAssment)

