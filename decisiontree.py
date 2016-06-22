#!/usr/bin/python

# Author: Lucas Tindall

from string import split
import math



def majority(data): 
    counts={}
    for row in data: 
        if(counts.has_key(row[-1])): 
            counts[row[-1]] += 1
        else: 
            counts[row[-1]] = 1
    majorityVote = ""
    maxCount = 0
    for key in counts.keys(): 
        if counts[key]>maxCount:
            maxCount = counts[key]
            majorityVote = key
    return majorityVote



def entropy(data): 
    rows = len(data)
    labelsCount = {}
    entropy = 0.0
    for row in data: 
        if(labelsCount.has_key(row[-1])): 
            labelsCount[row[-1]] += 1.0
        else: 
            labelsCount[row[-1]] = 1.0
    for key in labelsCount.keys(): 
        entropy += -(float(labelsCount[key])/len(data))*math.log(float(labelsCount[key])/len(data),2)
    return entropy

def pickFeatureSplit(data): 
    feature = 0
    greatestGain = 0.0
    numFeatures = len(data[0])-1
    entropyData = entropy(data)
    for numCol in range(numFeatures): 
	    valCounts = {}
            partialEntropy = 0.0
            for row in data: 
                if(valCounts.has_key(row[numCol])):
                    valCounts[row[numCol]] += 1.0
                else: 
                    valCounts[row[numCol]] = 1.0
            for value in valCounts.keys(): 
                probability = valCounts[value]/ float(len(data))
                matchingData = [row for row in data if row[numCol] == value]
		partialEntropy += probability*entropy(matchingData)
            gain = entropyData - partialEntropy
            if(gain > greatestGain): 
                greatestGain = gain
                feature = numCol
    return feature
     


def growTree(data, featureStrings): 
    labels = [row[-1] for row in data]
    if len(featureStrings) <= 0:
        return majority(data)
    elif labels.count(labels[0]) == len(labels): 
        return labels[0] 
    else:
	featureSplit = pickFeatureSplit(data)
	featureString = featureStrings[featureSplit]
        tree = {featureString:{}}
        featureColumn = [row[featureSplit] for row in data] 
	featureVals = set(featureColumn)
	for val in featureVals: 
	    copyFeatures = featureStrings[:]
	    copyFeatures.remove(featureString)
            newData = [[]]
	    for row in data: 
		if row[featureSplit] == val: 
		    newRow = []
		    for n in range(0,len(row)):
			if n != featureSplit: 
			    newRow.append(row[n])
		    newData.append(newRow)
	    newData.remove([])
            tree[featureString][val] = growTree(newData, copyFeatures)
	return tree

def test(tree, labels, testSet):
    testLabel = 0 
    firstSplit = tree.keys()[0]
    children = tree.values()[0]
    featureNum = labels.index(firstSplit)
    for child in children.keys():
	if testSet[featureNum] == child: 
	    if isinstance(children[child], dict): 
                testLabel = test(children[child], labels, testSet)
	    else:
   		testLabel = children[child]
  
    return testLabel 
 


def main(): 
    file = open("hw3train.txt")
    data = [[]]
    for line in file: 
        line = line.strip("\r\n")
        data.append(line.split(' '))
    file.close()
    data.remove([])
    for row in data: 
        del row[-1]
    featureStrings = ["petal width", "petal length", "sepal width", "sepal length"] 
    tree = growTree(data,featureStrings)
    print "Tree: "
    print tree
    file = open("hw3test.txt")
    testdata = [[]]
    for line in file: 
        line = line.strip("\r\n")
        testdata.append(line.split(' '))
    file.close()
    testdata.remove([])
    for row in testdata: 
        del row[-1]   
    error = 0
    for row in testdata: 
        prediction = test(tree, featureStrings, row)
        if prediction != row[-1]: 
            error +=1	
    print "Test Error: "
    print float(error)/len(testdata)

main()
