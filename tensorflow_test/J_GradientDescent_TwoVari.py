import numpy as np
from numpy import  genfromtxt
dataPath="house.csv"
dataSet=genfromtxt(dataPath,delimiter=',')
def getData(dataSet):
    m, n = np.shape(dataSet)
    trainData = np.ones((m, n))
    trainData[:,:-1]=dataSet[:,:-1]
    trainLabel=dataSet[:,-1]
    return  trainData,trainLabel
def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrains = x.transpose()
    for i in range(0, maxIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        gradient = np.dot(xTrains, loss) / m
        theta = theta - alpha * gradient
    return theta
trainData, trainLabel = getData(dataSet)
m, n = np.shape(trainData)
theta = np.ones(n)
alpha = 0.05
maxIteration = 10000
theta = batchGradientDescent(trainData, trainLabel, theta, alpha, m, maxIteration)
print(theta)
def predict(x, theta):
    m, n = np.shape(x)
    xTest = np.ones((m, n+1))
    xTest[:, :-1] = x
    yPre = np.dot(xTest, theta)
    return yPre
x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
print(predict(x, theta))