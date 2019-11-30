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
from sklearn.metrics import mean_squared_error

import multiprocessing
import timeit
from heapq import heapify,heappop,heappush
import random


def getDocLowestShingleID(featureVector):
	global randomNoA
	global randomNoB
	global k
	global prime
	lowestShingleID = []
	#print(docIndex)
	for x in range(0,k):
		listFx = []
		for i in range(len(featureVector)):
			if (featureVector[i] != 0):
				temp = ((randomNoA[x] * i + randomNoB[x]) % prime)%5100 # temp is a hashed value of a shingle of a doc.
				listFx.append(temp)
		heapify(listFx)
		lowestShingleID.append(heappop(listFx)) #lowestShingleID stores k lowest word's hashed index in the feature.
	return lowestShingleID

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


def findRandomNos(k,prime):
	randList = []
	while k>0:
		randIndex=random.randint(0, prime-1) 
		randList.append(randIndex)
		k = k-1
	
	return randList


k=16 #Number of hash functions. Must be modified here.

prime = 9551
randomNoA = [6327, 4253, 1056, 2475, 7633, 8034, 8714, 2973, 5590, 934, 5592, 837, 5400, 8964, 6838, 8628, 3613, 8411, 8387, 6715, 5003, 4108, 3825, 6318, 4698, 2502, 6358, 7958, 8279, 3313, 2127, 1056, 5546, 3481, 3226, 1641, 3261, 3605, 1100, 6050, 6222, 2832, 4197, 4999, 8885, 988, 5510, 1858, 4431, 2839, 7559, 3577, 3227, 9527, 2224, 1904, 2165, 6220, 4659, 2622, 4226, 6152, 2116, 3668, 6419, 2716, 3752, 214, 6382, 2125, 3998, 4102, 602, 6387, 4033, 5652, 2251, 8635, 8039, 3476, 8390, 7481, 7333, 7538, 3965, 1923, 9386, 9016, 5044, 5685, 2605, 5059, 1976, 2031, 9312, 8381, 5908, 3432, 582, 9171, 3528, 1842, 5700, 201, 5558, 4404, 5099, 2487, 1371, 8334, 5503, 3865, 1371, 2934, 3613, 2130, 3134, 6353, 8363, 8975, 9, 6170, 2271, 2725, 3559, 1889, 7583, 2336]
randomNoB = [2076, 6865, 7741, 4013, 7882, 7564, 8725, 1668, 7193, 5678, 5030, 1683, 5912, 3374, 7452, 951, 924, 20, 9099, 6597, 6385, 5264, 6058, 1637, 503, 4072, 1528, 2248, 3768, 1232, 6758, 356, 591, 548, 2839, 6415, 8105, 7883, 6719, 2267, 2456, 74, 434, 5236, 2316, 7366, 6322, 4545, 6343, 3374, 2100, 3000, 4569, 6271, 5712, 1897, 1242, 4929, 6947, 1882, 4189, 5946, 3387, 6034, 2049, 5355, 4796, 4353, 8883, 4020, 9355, 2093, 2148, 4160, 284, 2431, 8469, 2932, 986, 8246, 5257, 7841, 833, 2704, 1501, 1630, 7298, 3485, 6427, 6676, 1213, 8681, 6358, 8132, 1453, 3022, 2925, 2188, 8730, 6478, 4787, 8682, 4717, 9039, 1166, 3833, 1692, 4750, 4803, 1866, 3423, 6081, 3190, 5997, 7633, 45, 8607, 8719, 5614, 8912, 589, 8186, 9117, 7511, 6495, 3211, 6030, 5440]


"""-----------------------------Main Program-----------------------------------"""
if __name__ == '__main__':
	m = 3000 #data size
	n = 5100 #total words


	#Data preprocess
	print("Loading and processing data.")
	f = open(r'./data.txt', 'r',encoding = 'utf -8')
	labelVector = generatesLabelVector(f)
	f.seek(0)
	dirtyDict = GenerateDirtyDict(f, labelVector)
	print("The dictionary contains " + str(len(dirtyDict)+1)+" words.")
	f.seek(0)
	featureMatrix = generateFeatureMatrix(dirtyDict, n,f)
	print("FeatureMatrix is ready.")
	
	start = timeit.default_timer()
	dist=pairwise_distances(featureMatrix,metric='jaccard')


	sim = np.zeros([m,m], dtype = float)
	for i in range(sim.shape[0]):
		for j in range(sim.shape[1]):
			sim[i,j] = 1 - dist[i,j]

	print("Groundtruth Jaccard sim:")
	print(sim)
	end = timeit.default_timer()
	print('Time consumption by pairwise_distances function:', str(end-start),'s')


	#Reference:https://github.com/rahularora/MinHash/blob/master/minhash.py



	print("Please input mp, number of processes you want to run concurrently.")
	print("Recommended value is two times of CPU cores,")
	print("if you are not sure you can imput 4: ")
	mp = input()
	print("You chose mp to be:")
	print(mp)


	print("Calculating estimated sim. This will take one or two minutes according to k:")

	start = timeit.default_timer()
	p = multiprocessing.Pool(int(mp))
	docLowestShingleID = p.map(getDocLowestShingleID, featureMatrix)
	p.close()
	p.join()
	end1 = timeit.default_timer()
	print('Multi processing time:', str(end1-start),'s')
	print("Generating estimated sim.")
	estimateMatrix = []
	for x in range(0,3000):
		doc1LowestShingles = docLowestShingleID[x]
		col = []
		for y in range(0,3000):
			doc2LowestShingles = docLowestShingleID[y]
			count = 0
			for i in range(0,k):
				if doc1LowestShingles[i] == doc2LowestShingles[i]:
					count = count + 1
			col.append(count/k)
		estimateMatrix.append(col)
	npEstimateMatrix = np.asarray(estimateMatrix,dtype=np.float32)
	print("Estimated sim:")
	print(npEstimateMatrix)
	end2 = timeit.default_timer()
	print('Total time:', str(end2-start),'s')

	print("Calculating mean-squared error:")
	print(mean_squared_error(sim, npEstimateMatrix))
