# -*- coding: utf-8 -*-
from math import log
import matplotlib.pyplot as plt

# 计算数据集的熵  只是针对结果label而不是其他信息  默认最后一列为标签
def calcShannonEnt(dataSet):
    print '---shannonEnt:', dataSet
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries # 概率
        shannonEnt -= prob * log(prob, 2)       # 概率越高 熵越低
    print 'shannonEnt:',shannonEnt
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['not surfacing','flippers']
    return dataSet, labels
'''
mydata, labels = createDataSet()
print calcShannonEnt(mydata)   # 0.970950594455
mydata[0][-1]='mixedup'
print calcShannonEnt(mydata)  # 1.37095059445
# 概率降低 熵增加 越无序 熵越高
'''
# 给定特征划分数据集 dataSet[axis] = value
def splitDataSet(dataSet, axis, value):
    retDataSet =[]
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            #print reducedFeatVec
            reducedFeatVec.extend(featVec[axis+1:])
            #print reducedFeatVec
            retDataSet.append(reducedFeatVec)
    return retDataSet

'''
mydata, labels = createDataSet()
print mydata
print splitDataSet(mydata, 0, 0)
'''

# 针对每个特征获取特征个数然后计算总的熵  默认最后一列为标签
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) -1   # 
    bestFeature = 0
    baseEntropy = calcShannonEnt(dataSet)  # 计算总体熵
    bestInfoGain = 0.0; bsetFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        #print featList
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
           

mydata, labels = createDataSet()
print mydata
print chooseBestFeatureToSplit(mydata)


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] +=1
    sortedClassCount = sorted(classCount, key=lambda x:x[1], reverse = True)
    return sortedClassCount[0][0]

'''
遍历每个特征  从熵大的开始  也就是概率最低的 这样就将
这里核心思想是ID3而不是C4.5 俩者的相同点是节点选择时都是选择熵最大的
其子节点ID3不会和同级父节点重复, 而C4.5会根据情况会创建相同的 比如
ID3  人-(男,女) -> {（打球） ， （化妆）}
C4.5 人-(男,女) -> { （（打球），（化妆））,  （（打球），（化妆）） } 
'''
def createTrees(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    print '--1--',classList
    if classList.count(classList[0]) == len(classList): # 类别全一样停止划分
        return classList[0]
    print '---2----', dataSet[0]
    if len(dataSet[0]) == 1:   # 已结遍历了所有特征(除了标签)
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 计算当前特征中熵最大的特征
    bestFeatLabel = labels[bestFeat]            # 根据特征下标获取标签值
    print bestFeatLabel
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]  # 获取该特征全部特征值
    print '-----3----', featValues
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTrees(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
'''
mydata, labels = createDataSet()
print mydata
myTree = createTrees(mydata, labels)
print myTree
'''

# 使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel
'''
mydata, labels = createDataSet()
newLabels = labels[:]
myTree = createTrees(mydata, labels)
print classify(myTree, newLabels, [1,0])
''' 
    
    
    
    
    
    
    
    
    
    
