# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:42:40 2019

@author: Zhao,Zhiyuan
"""

from __future__ import division
import re
import numpy as np
import sys
from sklearn.metrics import pairwise_distances
import multiprocessing

import pickle
from heapq import heapify,heappop,heappush
import random

m = 3000 #data size
n = 5100 #total words
p = 460 #clean words
k = 5 #preserve word with frequency >=k, this is cleaning


def generatesLabelVector(f):
	contents = f.readline()
	labelVector = np.zeros(m, dtype = int)
	lineNumber = 0
	while (contents):
		labelVector[lineNumber] = int(contents[len(str(contents))-2])
		lineNumber +=1
		contents = f.readline()
	return labelVector

#While generating dirty dict, this also count pos/neg label nunbers for each word
def GenerateDirtyDict(f, labelVector):
	dirtyDict = {} 
	contents = f.readline()
	lineNumber = 0
	#global featureMatrix
	while (contents):
		lettersOnly = re.sub("[^a-zA-Z]", " ", contents)
		lineList = re.split(r'[.;,\s\t!?"\'/\-\\)(]', lettersOnly)
		for word in lineList:
			lowerWord = word.lower()
			if (lowerWord not in dirtyDict):
				dirtyDict[lowerWord] = [len(dirtyDict),0,0,0]
				#word:[word No., occurrences, positive, negative]
			dirtyDict[lowerWord][1] +=1
			if labelVector[lineNumber] == 1:
				dirtyDict[lowerWord][2] +=1
			else:
				dirtyDict[lowerWord][3] +=1
		lineNumber +=1
		contents = f.readline()
	return dirtyDict


#trivial words are like "the","an", they are frequent but not imformative
#We can assume for each word in a sentence, its label follows normal distribution.
#Thus it the ratio of pos/total is in or out of a confidence interval.
def createTrivialSet(dictionary, upperBoundofCI, lowerBoundofCI):
	trivialWordsSet = {'a'}
	for word in dictionary.keys():
		total = dictionary[word][2] + dictionary[word][3]
		pos = dictionary[word][2]
		if (pos/total <upperBoundofCI) and (pos/total >lowerBoundofCI):
			trivialWordsSet.add(word)
	return trivialWordsSet

def generateFeatureMatrix(dictToUse, w,f):
	featureMatrix = np.zeros([m,w], dtype = int)
	contents = f.readline()
	lineNumber = 0
	while (contents):
		lettersOnly = re.sub("[^a-zA-Z]", " ", contents)
		lineList = re.split(r'[.;,\s\t!?"\'/\-\\)(]', lettersOnly)
		for word in lineList:
			lowerWord = word.lower()
			if (lowerWord in dictToUse):
				featureMatrix[lineNumber][dictToUse[lowerWord][0]] +=1
		lineNumber +=1
		contents = f.readline()
	return featureMatrix

	
def generateCleanDict(trivialWordsSet, k,dirtyDict):
	dictionary = {}
	#global featureMatrix
	for word in dirtyDict.keys():
			lowerWord = word.lower()
			if (dirtyDict[lowerWord][1]>=k) and (lowerWord not in trivialWordsSet):
				dictionary[lowerWord] = [len(dictionary),0,0,0]
				dictionary[lowerWord][1] = dirtyDict[lowerWord][1]
				dictionary[lowerWord][2] = dirtyDict[lowerWord][2]
				dictionary[lowerWord][3] = dirtyDict[lowerWord][3]
				#word:[word No.]
	return dictionary

def printPredictedExamples(f,predictedLabel):
	contents = f.readline()
	lineNumber = 0
	while (lineNumber<5):
		print("A review and its groundtruth label:")
		print(contents + "Predicted label: "+ str(predictedLabel[lineNumber]))
		print(" ")
		lineNumber +=1
		contents = f.readline()

"""-----------------------------Main Program-----------------------------------"""

#Data preprocess
print("Loading and processing training data.")
f = open(r'./data.txt', 'r',encoding = 'utf -8')
labelVector = generatesLabelVector(f)
f.seek(0)
dirtyDict = GenerateDirtyDict(f, labelVector)
print("The uncleaned dictionary contains " + str(len(dirtyDict)+1)+" words.")
f.seek(0)
featureMatrix = generateFeatureMatrix(dirtyDict, n,f)
print("Training data is ready.")

"""
dist=pairwise_distances(featureMatrix,metric='jaccard')

sim = np.zeros([m,m], dtype = float)
for i in range(sim.shape[0]):
	for j in range(sim.shape[1]):
		sim[i,j] = 1 - dist[i,j]

print("Jaccard sim:")
print(sim)
"""
#Reference:https://github.com/rahularora/MinHash/blob/master/minhash.py


k = 25 #Number of hash functions.
prime = 5101

def findRandomNos(k,prime):
	randList = []
	randIndex = random.randint(0, prime -1) 
	randList.append(randIndex)
	while k>0:
		while randIndex in randList:
			randIndex = random.randint(0, prime-1) 
	
		randList.append(randIndex)
		k = k-1
	
	return randList

  
randomNoA = findRandomNos(k,prime)
randomNoB = findRandomNos(k,prime)

def getDocLowestShingleID(docIndex):
	global featureMatrix
	docLowestShingleID = {}
	lowestShingleID = []
	print(docIndex)
	for x in range(0,k):
		listFx = []
		for i in range(len(featureMatrix[docIndex])):
			if (featureMatrix[docIndex][i] != 0):
				temp = (randomNoA[x] * i + randomNoB[x]) % prime # temp is a hashed value of a word of a doc.
				listFx.append(temp)
		heapify(listFx)
		lowestShingleID.append(heappop(listFx)) #lowestShingleID stores k lowest word's hashed index in the feature.
	return lowestShingleID


p = multiprocessing.Pool(4)
docIndecs = [x for x in range(1, 10)]

docLowestShingleID = p.map(getDocLowestShingleID, docIndecs)
p.close()
p.join()


#for doc in docLowestShingleID:
#  print doc, docLowestShingleID[doc]

def getFileNo(x):
	if x<10:
		x = "0" + str(x)
	else:
		x = str(x)
	
	return x

estimateMatrix = []
for x in range(0,10):
	doc1LowestShingles = docLowestShingleID[x]
	col = []
	for y in range(0,10):
		doc2LowestShingles = docLowestShingleID[y]
		count = 0
		for i in range(0,k):
			if doc1LowestShingles[i] == doc2LowestShingles[i]:
				count = count + 1
	
		col.append(count/k)
	estimateMatrix.append(col)
npEstimateMatrix = np.asarray(estimateMatrix, dtype=np.float32)
print("Estimated sim:")
print(npEstimateMatrix)
