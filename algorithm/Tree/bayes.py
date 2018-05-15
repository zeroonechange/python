# -*- coding: utf-8 -*-
from numpy import *

# 过滤网站侮辱性文字
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]  # 1代表侮辱性文字
    return postingList, classVec

# 词集模型
def createVocabList(dataList):
    vocabSet = set([])
    for document in dataList:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 创建词向量   词汇表  文档
def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
            #returnVec[vocabList.index(word)] += 1  #词袋模型
        else:
            print('the world: %s is not in my vocabulary!' % word)
    return returnVec

'''
listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
print(myVocabList)
wordVec = setOfWord2Vec(myVocabList, listOPosts[0])
print(wordVec)
'''

# 分类器训练函数  params: 文档矩阵  每篇文档类别标签所构成的向量
def trainNBO(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)    # 非侮辱性个数
    p1Num = ones(numWords)    # 侮辱性个数
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)  # 去掉重复的
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
p0V, p1V, pAb = trainNBO(trainMat, listClasses)
print(p0V)
print(p1V)
print(pAb)   

# 分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    
    p0V, p1V, pAb = trainNBO(trainMat, listClasses)
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, 'classfied as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, 'classfied as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    
testingNB()