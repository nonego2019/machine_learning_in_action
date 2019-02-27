# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:24:55 2019

@author: Administrator
"""

import numpy as np
import pandas as pd
import operator


def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    print(classCount)
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
        
def file2matrix(filename):
    data = pd.read_table(filename,header=None)
    dataMatrix = data.values
    str={'smallDoses':2,'didntLike':1,'largeDoses':3}
    labelDf = data.iloc[:,-1].map(str)
    return dataMatrix[:,0:3],labelDf.values

def autonorm(dataset):
    minVals=dataset.min(0)
    maxVals=dataset.max(0)
    ranges = maxVals - minVals
    m=len(dataset)
    normDataset = dataset - np.tile(minVals,(m,1))
    normDataset = normDataset/np.tile(ranges,(m,1))
    return normDataset,ranges,minVals

def datingClassTest(k):
    hoRatio = 0.1
    DataMat,Labels = file2matrix('./datingTestSet.txt')
    normMat,ranges,minVals =autonorm(DataMat)
    m = len(normMat)
    #print(m)
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    #TestVecs = np.random.randint(0,m-1,size=numTestVecs)
    TestVecs = np.random.choice(m,size=numTestVecs,replace=False)#产生不重复的随机数
    #print(numTestVecs)
    train_data=np.zeros((m-numTestVecs,3))
    train_lebels=np.zeros(m-numTestVecs)
    test_data=np.zeros((numTestVecs,3))
    test_labels=np.zeros(numTestVecs)
    j=0
    s=0
    for i in range(m):
        if i not in TestVecs:
            #print(j)           
            train_data[j]=normMat[i]
            train_lebels[j]=Labels[i]
            j = j + 1
        if i in TestVecs:
            test_data[s]=normMat[i]
            test_labels[s]=Labels[i]
            s = s + 1
    for i in range(numTestVecs):        
        classifierResult = classify0(test_data[i],train_data,train_lebels,k)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, test_labels[i]))
        if (classifierResult != test_labels[i]):
            errorCount += 1
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))       
    
def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats = float(input("花费在视频游戏的时间比例"))
    ffMiles = float(input("每年的航空公里数"))
    iceCream = float(input("每年消费的冰淇淋公斤数"))
    datingDataMat,datingLabels=file2matrix('./datingTestSet.txt')
    normMat,ranges,minVals = autonorm(datingDataMat)
    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person:", resultList[classifierResult-1])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    