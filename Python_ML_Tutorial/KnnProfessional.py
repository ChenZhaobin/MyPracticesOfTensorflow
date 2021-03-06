"""
第一节课  初步认识KNN算法：
1.1 理论部分讲解
1.2 编写最简单的kNN代码
1.3 改进算法的性能  (使用numpy中的函数 np.array np.sum np.tile  np.argsort )
    为什么numpy的array那么快？https://www.zhihu.com/question/30823702
1.4 重构算法，使得代码符合机器学习的六个基本步骤
第二节课 写一个完整的KNN算法:
2.1  填充数据输入函数initDataSet()  主要是从文本文件中获取约会网站的数据,并将数据集分为训练样本和测试样本
2.2  填充预处理函数preProcessing()  主要讲解 归一化、样本权衡、特征不完整的处理(处理缺失值和异常值)等
     讲解脏数据产生的原因和预处理方法的分类 http://blog.csdn.net/u012162613/article/details/50629115
     讲解sklearn这个数据预处理工具包  http://blog.csdn.net/u012162613/article/details/50629115
2.3  填充训练算法函数coreTraining()，返回训练出来的model，并用日志记录训练过程和训练时间
2.4  填充测试算法函数 evaluate()
     主要讲解交叉验证和常见模型评价指标
     http://blog.csdn.net/losteng/article/details/50885057
     http://www.jianshu.com/p/6ffa3df3ec86
2.5 使用模型对新数据进行预测 predict()
第三节课 KNN设计思想提取(论文解读课)
第四节课 技能提升：如何用Python调用C语言编写的算法
4.1 用C语言实现一个KNN
4.2 Python调用C函数
4.3 思考多线程变量共享的问题【留到决策树章节去实现】
"""
from Python_ML_Tutorial.KnnModel import *
from numpy import *

def create_sample():
    """
    将文件中的数据转换成python可以处理的矩阵形式
    :return:
    """
    features = np.array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return features, labels


def pre_processing():
    """
    数据预处理 ，此处我们只对数据进行归一化
    """
    return


def core_training(training_set, training_labels, count_neighbor):
    """
    由于KNN算法没有训练的过程(也就是不能给出数据内在的含义)，
    所以我们直接将训练样本和训练标签作为训练模型的参数来使用。与其说KNN是算法，不如说是方法(Function)
    :param training_set: 训练数据集
    :param training_labels: 训练数据集对应的标签
    :param count_neighbor: 选取作为判断的邻居数目
    :return:
    """
    result_model = KnnModel(training_set, training_labels, count_neighbor)
    return result_model


def auto_norm(origin_dataset):
    min_vals = origin_dataset.min(0)
    max_vals = origin_dataset.max(0)
    ranges = max_vals-min_vals
    norm_dataset = np.zeros(np.shape(origin_dataset))  # 该变量初始化没什么意义，并没有用到
    m = origin_dataset.shape[0]
    norm_dataset = origin_dataset - np.tile(min_vals, (m, 1))
    norm_dataset = norm_dataset*(np.tile(ranges, (m, 1))**(-1))
    return norm_dataset, ranges, min_vals

def predict(model_knn, feature_vector_input):
    """

    :param model_knn:
    :param feature_vector_input:
    :return:
    """
    model_knn.predict(feature_vector_input)
    return


if __name__ == "__main__":
    sample_set, sample_labels = create_sample()
    print(auto_norm(sample_set))
    k = 3
    knnPreModel = core_training(sample_set, sample_labels, k)
    testX = np.array([0.1, 0.3])
    outputLabel = knnPreModel.predict(testX)
    print(outputLabel)
