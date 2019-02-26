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


    
    
    
    