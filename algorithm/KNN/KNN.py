# -*- coding: utf-8 -*-
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
import os

def createDataSet():
    group = array([[1.0, 1.1],[1.0,1.0],[0,0],[0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables
   
'''
拿每一个向量去和其他向量计算距离，得到一个最小的返回  非常耗时
k是用来选择最小的k个点
'''
def classify0(intX, dataSet, lables, k):
    dataSetSize = dataSet.shape[0]
   # print(dataSetSize)
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet
   # print(diffMat)
    sqDiffMat = diffMat**2
   # print(sqDiffMat)
    #[ 2.21  2.    0.    0.01]
    sqDistance = sqDiffMat.sum(axis=1)  # axis=1, 对每一行求和  axis=0,对每一列求和 
   # print(sqDistance)
    distance = sqDistance**0.5 
    sortedDistanceIndicies = distance.argsort() # 获取下标没有动源数据
    classCount = {}
    for i in range(k):
        voteIlabel = lables[sortedDistanceIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 # 对结果进行累加
        #print(classCount)
    #对累加结果进行排序得到最靠前的一个
    sortedClassCount = sorted(classCount, key=lambda x:x[0] , reverse=True)
    return sortedClassCount[0][0]
'''
dataSet, lables = createDataSet()
print(dataSet)
print(lables)
n = classify0([0,0], dataSet, lables, 3)
print(n)
'''

# 约会网站的配对  
def file2Matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberofLines = len(arrayOLines)
    returnMat = zeros((numberofLines, 3))
    classLableVector = [] # 最后一列
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 截取所有回车字符
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        #classLableVector.append(int(listFromLine[-1]))
        classLableVector.append(listFromLine[-1])
        index +=1
    return returnMat, classLableVector

# 归一化特征值  这样就解决了数据形式问题 newValue = oldValue - min / max - min 
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals 
    #print(minVals)
    #print(ranges)
    normDataSet = zeros(shape(dataSet))
    #print(normDataSet)
    m = dataSet.shape[0]  # 行数
    #print(m)
    normDataSet = dataSet - tile(minVals, (m, 1))  # tile 对行进行m次复制, 对列进行1次复制
    normDataSet = normDataSet / tile(ranges, (m , 1))
    return normDataSet, ranges, minVals
 
# 测试正确率
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2Matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classfierResule = classify0(normMat[i,:], 
                                    normMat[numTestVecs:m,:], datingLabels[numTestVecs:m]
                                    ,3)
        print('the classfier came back with: %s, the real answer is: %s' %(classfierResule, datingLabels[i]))
        if(classfierResule != datingLabels[i]): errorCount+=1.0
    print('the total error rate is: %f '%(errorCount/float(numTestVecs)))

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

'''
datingDataMat, datingLabels = file2Matrix('datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 
           15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()
normMat, ranges, minVals = autoNorm(datingDataMat)
#print(normMat)
#print(ranges)
#print(minVals)
'''
#datingClassTest()

'''
手写数字识别系统，其实就是将图像转化成一维数组然后算各个点的距离 效率太低了
'''
def handwritingClassTest():
    hwLabels = []
    traingFileList = os.listdir('digits/trainingDigits')
    m = len(traingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = traingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)  
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classfierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        if(classfierResult!=classNumStr): 
            errorCount+=1.0
            print('classfier : %s , the real : %s' %(classfierResult, classNumStr))
    print('\nthe total number of error is: %d' %errorCount)
    print('the train length:%d' % m)
    print('the test length:%d' % mTest)
    print('\nthe total error rate is: %f' %(errorCount/float(mTest)))
        
handwritingClassTest()
    

