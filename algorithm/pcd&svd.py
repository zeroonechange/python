from numpy import *
from numpy import linalg as la
def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]

Data = loadExData()
U, Sigma, VT = la.svd(Data)
#print U
print Sigma
'''
[  9.64365076e+00   5.29150262e+00   9.99338251e-16   4.38874654e-16
   1.19121230e-16]
 '''
#print VT
# 构建原始矩阵的 只使用前三个
Sig3 = mat([[Sigma[0], 0, 0],[0, Sigma[1], 0],[0, 0, Sigma[2]]])
temp = U[:,:3]*Sig3*VT[:3,:]
print temp
但是为什么组装不起来原来的矩阵呢? 


import numpy as np
b= np.mat(np.arange(20).reshape(4,5))
print(b)
print(b[1:3,2:5])   # 先取第一维中下标为1,2的2部分，再取第二维中下标为2,3,4的3部分
print(b[:2,2:])     # 同理，前面不写从头开始，后面不写一直到末尾
print(b[:2,3])      # 当然，也可以在某维度上只取一行
对矩阵进行切片就是行和列的俩个集合表示, 中间用,号隔开


from numpy import *
from numpy import linalg as la

# 欧式距离的计算 (0,1]
def euliSin(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))
# 皮尔逊相关系数 (0,1]
def pearsSim(inA, inB):
    if len(inA) < 3 : return 1.0
    return 0.5 + 0.5*corrcoef(inA, inB, rowvar=0)[0][1]
# 余弦相似度 (0,1]
def cosSim(inA, inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5*(num/denom)

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
myMat = mat(loadExData())
print euliSin(myMat[:,0], myMat[:,4]) #0.129731907557
print euliSin(myMat[:,0], myMat[:,0]) #1.0

print pearsSim(myMat[:,0], myMat[:,4]) #0.205965381738
print pearsSim(myMat[:,0], myMat[:,0]) #1.0

print cosSim(myMat[:,0], myMat[:,4]) #0.5
print cosSim(myMat[:,0], myMat[:,0]) #1.0
不知道三者哪个最准确，处理起来速度最快

print myMat[2, :]  # [[4 0 0 1 1]]
print nonzero(myMat[2, :].A==0) #(array([0, 0], dtype=int64), array([1, 2], dtype=int64))
print nonzero(myMat[2, :].A!=0) #(array([0, 0, 0], dtype=int64), array([0, 3, 4], dtype=int64))
for j in range(n):
    overLap = nonzero(logical_and(myMat[:, 2].A > 0, myMat[:,j].A > 0)) # 相反的 (array([3, 4, 5, 6], dtype=int64), array([0, 0, 0, 0], dtype=int64)) 
    print overLap

print shape(myMat) # 7行5列的矩阵  shape(dataMat)[1]=5 (7L, 5L) 

from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
# 欧式距离的计算 (0,1]
def euliSin(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))
# 皮尔逊相关系数 (0,1]
def pearsSim(inA, inB):
    if len(inA) < 3 : return 1.0
    return 0.5 + 0.5*corrcoef(inA, inB, rowvar=0)[0][1]
# 余弦相似度 (0,1]
def cosSim(inA, inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5*(num/denom)

# 计算在给定相似度计算方法的条件下用户对物品的估计评分值
# dataMat：原始数据  user：用户下标(行数) item:未评分的菜式列下标
def standEst(dataMat, user, simMeans, item):
    n = shape(dataMat)[1]  
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0: continue
        # 集合{列item值>0}  ∩  集合{列j值>0}  交集
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:,j].A > 0))[0]
        if len(overLap) == 0: similarity = 0
        # 计算俩列的相关性
        else: similarity = simMeans(dataMat[overLap, item], dataMat[overLap, j])
        #print 'the %d and %d similarity is : %f' %(item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
    
def recommend(dataMat, user, N=3, simMeans = cosSim, estMethod = standEst):
    # nonzero 返回是不为0的下标
    unreatedItems = nonzero(dataMat[user, :].A==0)[1]
    #print unreatedItems  # 未评价的菜式列下标集合[1,2] 
    if len(unreatedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unreatedItems:
        estimatedScore = estMethod(dataMat, user, simMeans, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key = lambda jj: jj[1], reverse=True)[:N]

myMat = mat(loadExData())
myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
print recommend(myMat, 2)
#print recommend(myMat, 2, simMeans=euliSin)
#print recommend(myMat, 2, simMeans=pearsSim)


def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

'''
U, Sigma, VT = la.svd(mat(loadExData2()))
print len(Sigma) # 11维
print Sigma  
Sig2 = Sigma**2  # 求平方
sumS = sum(Sig2)
print sumS      # 542.0
standard = sumS*0.9  # 达到总能量的90%就可以对矩阵降维
print sum(Sig2[:2]) # 378.829559511 / 542.0 = 0.698947526774
print sum(Sig2[:3]) # 500.500289128 / 542.0 = 0.923432267763
# 将11维矩阵转化成3维的矩阵
'''
def svdEst(dataMat, user, simMeans, item):
    n = shape(dataMat)[1]  
    simTotal = 0.0; ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4])  # 建立对角矩阵
    print '-----before--------------'
    print Sig4
    print dataMat.T  # transpose of the matrix 转置
    print U[:,:4]
    print Sig4.I  # inverse of invertible self  逆
    print '-----end--------------'
    xformedItems = dataMat.T * U[:,:4] * Sig4.I # 构建转换后的物品（从11维到4维）
    #print xformedItems
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item: continue
        similarity = simMeans(xformedItems[item,:].T, xformedItems[j,:].T)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

myMat = mat(loadExData2())
print recommend(myMat, 1, estMethod=svdEst)
#print recommend(myMat, 1, estMethod=svdEst, simMeans=pearsSim)
#print recommend(myMat, 1, estMethod=svdEst, simMeans=euliSin)
'''
[(4, 3.3447149384692283), (7, 3.3294020724526963), (9, 3.328100876390069)]
[(4, 3.3469521867021736), (9, 3.3353796573274699), (6, 3.307193027813037)]
[(4, 3.3286756747000452), (9, 3.3247038080937834), (7, 3.3224884985810177)]
'''

def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print 1,
            else:
                print 0,
        print ''
'''
图像压缩  从 32X32=1024 压缩到32X2 + 2 + 32X2 = 130  1024/130=7.876
numSV： SVD分类个数 有用的维度个数
thresh： 阈值
'''      
def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print '----------original matrix-----------'
    printMat(myMat, thresh)
    U, Sigma, VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))   
    for k in range(numSV):
        SigRecon[k,k] = Sigma[k]  # 将Sigma重构成SigRecon
    reconMat = U[:,:numSV] * SigRecon*VT[:numSV,:]  # 根据三个小矩阵还原至原矩阵
    print '------------reconstructed matrix using %d singular values-------' % numSV
    printMat(reconMat, thresh)

imgCompress(2) 
		   
		   
	
********************************* PCA ***********************************************
1000L  1L的数据进行降维后画图解析
from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat = 9999999):
    meanVals = mean(dataMat, axis = 0)          # 求平均值   [[ 9.06393644  9.09600218]]
    meanRemoved =  dataMat - meanVals           # 均值
    covMat = cov(meanRemoved, rowvar = 0)       # 构建协方差矩阵 
    eigVals, eigVects = linalg.eig(mat(covMat)) # 特征值 特征向量
    ''' 
    eigVals = [ 0.36651371  2.89713496]
    eigVects = [[-0.85389096 -0.52045195]
                [ 0.52045195 -0.85389096]] 
    '''
    eigValInd = argsort(eigVals)         # 给特征值从小到大排序   [0, 1]
    eigValInd = eigValInd[:-(topNfeat + 1): -1] # 最大个数特征值  [1]
    redEigVects = eigVects[:,eigValInd]          # 保留前N个特征向量
    lowDDataMat = meanRemoved * redEigVects      # 将数据转换到N个特征向量构建的新空间中
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 在新特征值下还原至原数据
    return lowDDataMat, reconMat

dataMat = loadDataSet('testSet.txt')
lowDMat, reconMat = pca(dataMat, 1)
print shape(lowDMat)
print lowDMat
print reconMat

import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90)
ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=50, c='red')

		   
对于缺省值NaN用平均值来代替   590个特征5.3M  ML用于测试发现有缺陷的产品  
def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')  
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0], i])  # 计算所有非NaN的平均值
        datMat[nonzero(isnan(datMat[:,i].A))[0], i] = meanVal   # 将所有NaN置为平均值 
    return datMat
dataMat = replaceNanWithMean()
# print shape(dataMat) # 1567L 590L
meanVals = mean(dataMat, axis =0)  # 获取均值
meanRemoved = dataMat - meanVals   # 去均值
covMat = cov(meanRemoved, rowvar = 0) # 协方差矩阵
eigVals, eigVects = linalg.eig(mat(covMat))
#print eigVals   #前20个主成分覆盖了99.3%的方差
for i in range(len(eigVals)*2/10):
    print eigVals[i]



