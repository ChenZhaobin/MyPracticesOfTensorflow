"""

参考：http://blog.csdn.net/zouxy09/article/details/16955347
最近邻算法(kNN: k Nearest Neighbors)
输入:    dataSet：数据集(NxM)【M个训练样本，每个样本具有N个属性】
         labels: 数据集相对应的标签(1xM vector)
         newInput：需要和已有数据集进行比较的输入样本(1xN)【一行N列】
         k: 用来做判断的邻居数目
输出:   prop_label 最可能的分类标签
Python 是双面向的,既可以面向函数编程,也可以面向对象编程,所谓面向函数就是单独一个. py 文件,里面没有类,全是一些函数,调用的时候导入模块,通过模块名.函数名()即可调用,完全不需要类,那么你可能会问,那要类还有什么毛用? 类就是用来面向对象编程啦,类可以有自己的属性,类可以创建很多实例,每个实例可以有不同的属性,这也就保存了很多私有的数据,总之都有存在的必要.

"""
import numpy as np
def createDataSet():
    group = np.array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels=['A', 'A', 'B', 'B']
    return group,labels
def kNNClassify(dataSet,labels,newInput,k):
    max_label=labels[0]
    numSamples=dataSet.shape[0]
    repeatInput= np.tile(newInput,(numSamples,1))
    diff=repeatInput-dataSet
    squaredDiff=diff**2
    squaredDist = np.sum(squaredDiff, axis=1)
    distance = squaredDist ** 0.5
    sortedDistIndices = np.argsort(distance)
    classCount={}
    for i in range(k):
       cur_index=sortedDistIndices[i]
       cur_label=labels[cur_index]
       classCount[cur_label]=classCount.get(cur_label,0)+1
    maxCount=0
    for key,value in classCount.items():
        if value > maxCount:
            maxCount = value
            max_label=key
    return max_label
dataSet, labels = createDataSet()
testX = np.array([0.1, 0.3])
k = 3
outputLabel = kNNClassify(dataSet, labels, testX, 3)
print(outputLabel)
