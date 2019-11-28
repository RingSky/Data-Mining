# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:42:40 2019

@author: Zhao,Zhiyuan
"""

from __future__ import division
import re
import numpy as np
import sys


m = 2101 #data size
n = 4100 #total words
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
f = open(r'./trainingData.txt', 'r',encoding = 'utf -8')
labelVector = generatesLabelVector(f)
f.seek(0)
dirtyDict = GenerateDirtyDict(f, labelVector)
trivialWordsSet = createTrivialSet(dirtyDict, 0.6, 0.4)
dictionary = generateCleanDict(trivialWordsSet,k,dirtyDict)
print("The uncleaned dictionary contains " + str(len(dirtyDict)+1)+" words.")
print("The cleaned dictionary contains " + str(len(dictionary)+1)+" words.")
f.seek(0)
cleanFeatureMatrix = generateFeatureMatrix(dictionary, p,f)
f.seek(0)
dirtyFeatureMatrix = generateFeatureMatrix(dirtyDict, n,f)
print("Training data is ready.")



#predict on test set
dirtyData = 0
if (str(sys.argv[1])) == "dirty":
	dirtyData = 1
m = 450 #data size
print("Loading and preprocessing test data.")
f = open(r'./testSet.txt', 'r',encoding = 'utf -8')
labelVectorVal = generatesLabelVector(f)
f.seek(0)
cleanFeatureMatrixVal = generateFeatureMatrix(dictionary, p,f)
f.seek(0)
dirtyFeatureMatrixVal = generateFeatureMatrix(dirtyDict, n,f)
print("Test data is ready.")
from sklearn import tree  #Using decision tree
clf = tree.DecisionTreeClassifier()
if dirtyData == 1:
	print("Training decision tree model with uncleaned data.")
	clf = clf.fit(dirtyFeatureMatrix, labelVector)
	predictedLabel = clf.predict(dirtyFeatureMatrixVal)
	print("Test accuracy is " +str(np.mean((labelVectorVal == predictedLabel)) * 100)+"%.")
else:
	print("Training decision tree model with clean data.")
	clf = clf.fit(cleanFeatureMatrix, labelVector)
	predictedLabel = clf.predict(cleanFeatureMatrixVal)
	print("Test accuracy is " +str(np.mean((labelVectorVal == predictedLabel)) * 100)+"%.")


#predict on val set
print("Loading and preprocessing val data.")
f = open(r'./valSet.txt', 'r',encoding = 'utf -8')
labelVectorVal = generatesLabelVector(f)
f.seek(0)
cleanFeatureMatrixVal = generateFeatureMatrix(dictionary, p,f)
f.seek(0)
dirtyFeatureMatrixVal = generateFeatureMatrix(dirtyDict, n,f)
print("Val data is ready.")
if dirtyData == 1:
	predictedLabel = clf.predict(dirtyFeatureMatrixVal)
	print("Val accuracy is " +str(np.mean((labelVectorVal == predictedLabel)) * 100)+"%.")
else:
	predictedLabel = clf.predict(cleanFeatureMatrixVal)
	print("Val accuracy is " +str(np.mean((labelVectorVal == predictedLabel)) * 100)+"%.")

#predict on training data
if dirtyData == 1:
	predictedLabel = clf.predict(dirtyFeatureMatrix)
	print("Training accuracy is " +str(np.mean((labelVector == predictedLabel)) * 100)+"%.")
else:
	predictedLabel = clf.predict(cleanFeatureMatrix)
	print("Training accuracy is " +str(np.mean((labelVector == predictedLabel)) * 100)+"%.")


#print examples
print("Examples:")
f = open(r'./testSet.txt', 'r',encoding = 'utf -8')
printPredictedExamples(f,predictedLabel)
















