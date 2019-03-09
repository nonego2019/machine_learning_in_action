# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:47:06 2019

@author: Administrator
"""

import numpy as np
import re
import random
import operator
import feedparser


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]  
    return postingList,classVec

#汇总词表列表成包含所有单词的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):   
    #inputSet为单个文档组成的词条向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('单词:% 没有出现在词汇表中' % word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])  #单词的数量，也就是词汇表的长度
    pAbusive = float(sum(trainCategory)/numTrainDocs)
    
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    #p0Denom = 0
    #p1Denom = 0
    p0Denom = 2
    p1Denom = 2 #每个特征的取值有两个所以2*1=2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i]) 
            
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    #类型是1的每个文档中单词种类数的和，含义不是很有意义，计算概率方法和公式不同
    #按照公式此值应该取sum(trainCategory),如下代码
    #p1Vect = np.log(p1Num/(sum(trainCategory)+2))
    #p0Vect = np.log(p0Num/(numTrainDocs-sum(trainCategory)+2))
    print(p1Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + np.log(1-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc)) #将训练文档转成词向量
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry)) #将测试文档转成向量
    print(testEntry,'分类:',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'分类:',classifyNB(thisDoc,p0V,p1V,pAb))
    
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1   
           
    return returnVec

def textParse(bigString):
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):#文件名从1开始
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)  #词汇没有去重
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList) #使用fullText去重后应该也可以
    trainingSet = list(range(50))
    testSet =[]
    for i in range(10):
        randIndex = random.randint(0,len(trainingSet)-1)
        print(randIndex)
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(trainMat,trainClasses)
    errorCount = 0
    
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount += 1
    print('错误率是',errorCount/len(testSet))
    
def calcMostFreq(vocabList,fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(),key = operator.itemgetter(1),reverse=True);
    return sortedFreq[:30]

def localWord(feed1,feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(int(minLen/10)):            #生产测试集、训练集在docList中的索引
        randIndex = random.randint(0,len(trainingSet)-1)
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount += 1
    print('错误率是',errorCount/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    vocabList,p0V,p1V = localWord(ny,sf)
    topNY=[]
    topSF=[]
    for i in range(len(p0V)):
        if p0V[i]>-6.0:
            topNY.append((vocabList[i],p0V[i]))
    for i in range(len(p1V)):
        if p1V[i]>-6.0:
            topSF.append((vocabList[i],p1V[i]))     
    sortedSF = sorted(topSF,key=lambda pair:pair[1],reverse=True)   
    
    print ("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0],item[1])
    sortedNY = sorted(topNY,key=lambda pair:pair[1],reverse=True)   
    
    print ("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedNY:
        print(item[0],item[1])
            
        
        
    
        


        
    
    
    
    
    








































