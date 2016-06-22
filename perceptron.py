#!/usr/bin/python

# Author: Lucas Tindall

import numpy


def isClass( a, b): 
  if a == b: 
    return 1
  else: 
    return -1

def perceptron( data, passes): 
 
  w = numpy.zeros((1,784))
  m = 0
  for j in range(passes):
    for i in range(len(data)): 
      if isClass(0,data[i,-1]) * numpy.dot(w[m],data[i,0:-1]) <= 0: 
        w = numpy.concatenate((w,numpy.zeros((1,784))))
        w[m+1] = w[m] + isClass(0,data[i,-1]) * data[i,0:-1]
        m = m+1
    #print "%d run: w[%d]:" % (j+1,m),  
   # print w[m]
  #return numpy.sign(numpy.dot(w[m],test[0:-1]))
  return w[m]

def perceptronClass( data, passes, label): 
 
  w = numpy.zeros((1,784))
  m = 0
  for j in range(passes):
    for i in range(len(data)): 
      if isClass(label,data[i,-1]) * numpy.dot(w[m],data[i,0:-1]) <= 0: 
        w = numpy.concatenate((w,numpy.zeros((1,784))))
        w[m+1] = w[m] + isClass(label,data[i,-1]) * data[i,0:-1]
        m = m+1
    #print "%d run: w[%d]:" % (j+1,m),  
   # print w[m]
  #return numpy.sign(numpy.dot(w[m],test[0:-1]))
  return w[m]

def voteAvgPerceptron( data, passes): 
 
  w = numpy.zeros((1,784))
  m = 0
  c = numpy.ones((1,1))
  for j in range(passes):
    for i in range(len(data)): 
      if isClass(0,data[i,-1]) * numpy.dot(w[m],data[i,0:-1]) <= 0: 
        w = numpy.concatenate((w,numpy.zeros((1,784))))
        w[m+1] = w[m] + isClass(0,data[i,-1]) * data[i,0:-1]
        m = m+1
        c = numpy.concatenate((c,numpy.ones((1,1))))
      else: 
        c[m] = c[m] +1

       
    #print "%d run: w[%d]:" % (j+1,m),  
    #print w[m]
  #return numpy.sign(numpy.dot(w[m],test[0:-1]))
  return w,c

def main(): 
  trainingData = numpy.genfromtxt('hw4atrain.txt', delimiter=" ")
  testData = numpy.genfromtxt('hw4atest.txt', delimiter=" ")

  trainingDataB = numpy.genfromtxt('hw4btrain.txt', delimiter=" ")
  testDataB = numpy.genfromtxt('hw4btest.txt', delimiter=" ")
 
  #Training Data, Regular Perceptron  
  for j in range(3):
    w = perceptron(trainingData, j+1 )
    trainingError = 0 
    for i in range(len(trainingData)): 
      if( numpy.sign(numpy.dot(w, trainingData[i,0:-1])) != isClass(0,trainingData[i,-1])): 
        trainingError +=1
    print "Pass#: %d, Perceptron Training Error: %f " % (j+1,float(trainingError)/len(trainingData)) 
  
  print ""    
  #Training Data, Voted Perceptron, Avg Perceptron 
  avgErrors = numpy.zeros((3,1))
  voteErrors = numpy.zeros((3,1))
  for j in range(3): 
    wV,cV = voteAvgPerceptron(trainingData,j+1) 
    trainingErrorAvg = 0
    trainingErrorVote = 0
    for i in range(len(trainingData)): 
      avgSigma = 0
      for n in range(len(wV)): 
        avgSigma = avgSigma + cV[n]*wV[n]
      if( numpy.sign(numpy.dot(avgSigma, trainingData[i,0:-1] )) != isClass(0,trainingData[i,-1]) ): 
        trainingErrorAvg += 1
      voteSigma = 0
      for n in range(len(wV)): 
        voteSigma += cV[n] * numpy.sign(numpy.dot(wV[n], trainingData[i,0:-1]))
      if(numpy.sign(voteSigma) != isClass(0, trainingData[i,-1])): 
        trainingErrorVote += 1
    avgErrors[j] = trainingErrorAvg
    voteErrors[j] = trainingErrorVote
    #print "Pass#: %d, Average Perceptron Training Error: %d " % (j+1, trainingErrorAvg)
    #print "Pass#: %d, Voted Perceptron Training Error: %d" % (j+1, trainingErrorVote)

  for i in range(3): 
    print "Pass#: %d, Average Perceptron Training Error: %f" % (i+1, float(avgErrors[i])/len(trainingData))
  print ""
  for i in range(3): 
    print "Pass#: %d, Voted Perceptron Training Error: %f" % (i+1, float(voteErrors[i])/len(trainingData)) 

  print "" 

  #Test Data, Regular Perceptron  
  for j in range(3):
    wTest = perceptron(trainingData, j+1 )
    testError = 0 
    for i in range(len(testData)): 
      if( numpy.sign(numpy.dot(wTest, testData[i,0:-1])) != isClass(0,testData[i,-1])):  
       testError +=1
    print "Pass#: %d, Perceptron Test Error: %f " % (j+1,float(testError)/len(testData)) 
  
  print ""    
  #Test Data, Voted Perceptron, Avg Perceptron 
  avgErrors = numpy.zeros((3,1))
  voteErrors = numpy.zeros((3,1))
  for j in range(3): 
    wV,cV = voteAvgPerceptron(trainingData,j+1) 
    testErrorAvg = 0
    testErrorVote = 0
    for i in range(len(testData)): 
      avgSigma = 0
      for n in range(len(wV)): 
        avgSigma += cV[n]*wV[n]
      if( numpy.sign(numpy.dot(avgSigma, testData[i,0:-1] )) != isClass(0,testData[i,-1]) ): 
        testErrorAvg += 1
      voteSigma = 0
      for n in range(len(wV)): 
        voteSigma += cV[n] * numpy.sign(numpy.dot(wV[n], testData[i,0:-1]))
      if(numpy.sign(voteSigma) != isClass(0, testData[i,-1])): 
        testErrorVote += 1
    avgErrors[j] = testErrorAvg
    voteErrors[j] = testErrorVote
    #print "Pass#: %d, Average Perceptron Training Error: %d " % (j+1, trainingErrorAvg)
    #print "Pass#: %d, Voted Perceptron Training Error: %d" % (j+1, trainingErrorVote)

  for i in range(3): 
    print "Pass#: %d, Average Perceptron Test Error: %f" % (i+1, float(avgErrors[i])/len(testData))
  print ""
  for i in range(3): 
    print "Pass#: %d, Voted Perceptron Test Error: %f" % (i+1, float(voteErrors[i])/len(testData)) 



  # one vs. all  
  c = numpy.zeros((10,784))
  for j in range(10):
    c[j] = perceptronClass(trainingDataB, 1, j)
   
  
  predictions = 10*numpy.ones(len(testDataB))  
  for i in range(len(testDataB)): 
    prediction = 0
    label = 10
    for j in range(10): 
      if( numpy.sign(numpy.dot(c[j], testDataB[i,0:-1])) == 1 ):  
        #print numpy.sign(numpy.dot(c[j], testDataB[i,0:-1])), " ", float(isClass(j, testDataB[i,-1])) 
        prediction += 1
        label = j 
 
    if(prediction == 1): 
      predictions[i] = label 

  
  confusion = numpy.zeros((11,10)) 
  labelCounts = numpy.zeros(10) 
  for i in range(len(testDataB)): 
    confusion[int(predictions[i])][int(testDataB[i,-1])] += 1
    labelCounts[testDataB[i,-1]] += 1

  print "" 
  print "         ",
  for i in range(len(labelCounts)): 
    confusion[:,i] /= labelCounts[i]
    print "%06d" %(i), 
  print "" 
  print "      ", 
  for j in range(35): 
    print "-", 
  print ""
  for j in range(len(labelCounts)+1): 
    print " %05d" %(j), " |", 
    for i in range(len(labelCounts)):
      print "%.4f" %(confusion[j][i]), 
    print "" 
     
main()
