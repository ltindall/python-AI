#!/usr/bin/python

#Author: Lucas Tindall
#CSE 151 - PS 2
import numpy 


def getKNNLabel(trainingData, testVector, k): 
    distanceAndLabels = []
    for i in range(len(trainingData)):
        distance = numpy.linalg.norm(testVector[0:-1] - trainingData[i,0:-1])
        distanceAndLabels.append([distance,trainingData[i,-1]]); 
    distanceAndLabels.sort(key = lambda x: int(x[0]))
    kNearestNeighbors = []
    labelsCount = {}
    for i in range(k):
        kNearestNeighbors.append(distanceAndLabels[i][1])
	tempLabel = distanceAndLabels[i][1] 
        #print("tempLabel ")
        #print(tempLabel)
        if tempLabel in labelsCount: 
            labelsCount[tempLabel] += 1
        else: 
            labelsCount[tempLabel] = 1
    return max(labelsCount, key=labelsCount.get)




    

def main(): 
    trainingData = numpy.genfromtxt('hw2train.txt', delimiter=" ") 
    validationData = numpy.genfromtxt('hw2validate.txt', delimiter=" ")
    testData = numpy.genfromtxt('hw2test.txt', delimiter=" ")

    k = [1,3,5,11,16,21]
    trainingErrors = []
    validationErrors = []
    testErrors = []
    for i in range(len(k)):
        trainingErrorCount = 0 
        for j in range(len(trainingData)): 
            label = getKNNLabel(trainingData, trainingData[j], k[i])
            if(label != trainingData[j,-1]): 
                trainingErrorCount += 1
        trainingErrors.append([k[i],trainingErrorCount])
        print "Training Error for k = ", k[i],": ",float(trainingErrorCount)/len(trainingData)
        validationErrorCount = 0 
        for j in range(len(validationData)): 
            label = getKNNLabel(trainingData, validationData[j], k[i])
            if(label != validationData[j,-1]): 
                validationErrorCount += 1
        validationErrors.append([k[i],validationErrorCount]) 
        print "Validation Error for k = ",k[i],": ",float(validationErrorCount)/len(validationData)

        testErrorCount = 0 
        for j in range(len(testData)): 
            label = getKNNLabel(trainingData, testData[j], k[i])
            if(label != testData[j,-1]): 
                testErrorCount += 1
        testErrors.append([k[i],testErrorCount])
        print "Test Error for k = ",k[i],": ",float(testErrorCount)/len(testData)


    confusionMatrix = numpy.zeros((10,10))
    labelCounts = numpy.zeros(10) 
    for j in range(len(testData)): 
	label = getKNNLabel(trainingData, testData[j], 3)
	confusionMatrix[label][testData[j,-1]] += 1
	labelCounts[testData[j,-1]] += 1
    print""
    print "         ",
    for j in range(len(labelCounts)):
        confusionMatrix[:,j]/=labelCounts[j]
	print "%06d" %(j),
    print""
    print"         ",
    for j in range(35):
        print"-",
    print""       
    for j in range(len(labelCounts)):
        print " %05d" %(j), " |",
        for n in range(len(labelCounts)):
	     print "%.4f" %(confusionMatrix[j][n]),
        print ""


main()




       

