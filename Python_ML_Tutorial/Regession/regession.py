from numpy import *


def load_dataset(filename):
    features = []
    labels = []
    with open(filename) as fr:
        for line in fr.readlines():
            tokens = line.strip().split("\t")
            features.append([float(tk) for tk in tokens[:-1]])
            labels.append(float(tokens[-1]))
    return features, labels


def regress_standard(sample_features, sample_labels):
    mat_features = mat(sample_features)
    mat_labels = mat(sample_labels).T
    featuretrans_mult_feature = mat_features.T * mat_features
    if linalg.det(featuretrans_mult_feature) == 0.0:
        print("This matrix is singular , cannot do inverse")
        return
    return featuretrans_mult_feature.I * (mat_features.T * mat_labels)

xArr, yArr = load_dataset("ex0.txt")
ws = regress_standard(xArr, yArr)
print("ws（回归系数）："+str(ws))
#
#
# def show():
#     import matplotlib.pyplot as plt
#     xMat = mat(xArr)b
#     yMat = mat(yArr)
#     fig = plt.figure() #创建绘图对象
#     ax = fig.add_subplot(111)  #111表示将画布划分为1行2列选择使用从上到下第一块
#     #scatter绘制散点图
#     ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
#     #复制，排序
#     xCopy =xMat.copy()
#     xCopy.sort(0)
#     yHat = xCopy * ws
#     #plot画线
#     ax.plot(xCopy[:,1],yHat)
#     plt.show()
#
# show()

#利用numpy库提供的corrcoef来计算预测值和真实值得相关性
# yHat = mat(xArr) * ws  #yHat = xMat * ws
# print("相关性："+str(corrcoef(yHat.T,mat(yArr))))
#====================用线性回归找到最佳拟合曲线===========


#==================局部加权线性回归================

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))   #产生对角线矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        #更新权重值，以指数级递减
        weights[j,j] = exp(diffMat * diffMat.T /(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("this matrix is singular,cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] =lwlr(testArr[i],xArr,yArr,k)
    return yHat


xArr,yArr = load_dataset('ex0.txt')
print("k=1.0：",lwlr(xArr[0],xArr,yArr,1.0))
print("k=0.001：",lwlr(xArr[0],xArr,yArr,0.001))
print("k=0.003：",lwlr(xArr[0],xArr,yArr,0.003))

#画图
def showlwlr():
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    xMat = mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    
    import matplotlib.pyplot as plt
    fig = plt.figure() #创建绘图对象
    ax = fig.add_subplot(111)  #111表示将画布划分为1行2列选择使用从上到下第一块
    ax.plot(xSort[:,1],yHat[srtInd])
    #scatter绘制散点图
    ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T[:,0].flatten().A[0],s=2,c='red')
    plt.show()

showlwlr()

'''
#=========================岭回归==================
#用于计算回归系数
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom)==0.0:
        print "This matrix is singular, cannot do inverse"
        return 
    ws = denom.I * (xMat.T * yMat)
    return ws

#用于在一组lambda上做测试
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat,0)
    #数据标准化
    yMat = yMat - yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMeans)/xVar
    
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i,:]=ws.T
    return wMat

abX,abY = loadDataSet('abalone.txt')
ridgeWeights = ridgeTest(abX,abY)
# print ridgeWeights

def showRidge():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

showRidge()
#===================岭回归=============
'''
#===================向前逐步回归============

#计算平方误差
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

#数据标准化处理
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):#could change this to while loop
        #print ws.T
        lowestError = inf;
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

xArr,yArr = load_dataset('abalone.txt')
print(str(stageWise(xArr, yArr, 0.01, 200))+"\n\n")

# print stageWise(xArr, yArr, 0.001, 200)

xMat = mat(xArr)
yMat = mat(yArr).T
xMat = regularize(xMat)
yM = mean(yMat,0)
yMat = yMat - yM
weights = regress_standard(xMat, yMat.T)
print(weights.T)