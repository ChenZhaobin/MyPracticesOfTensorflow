import cv2
import numpy as np
import math
import sys
import time
def stretchImage(data, s=0.005, bins=2000):  # 线性拉伸，去掉最大最小0.5%的像素值，然后线性拉伸至[0,1]
    ht = np.histogram(data, bins);
    d = np.cumsum(ht[0]) / float(data.size)
    lmin = 0;
    lmax = bins - 1
    while lmin < bins:
        if d[lmin] >= s:
            break
        lmin += 1
    while lmax >= 0:
        if d[lmax] <= 1 - s:
            break
        lmax -= 1
    return np.clip((data - ht[1][lmin]) / (ht[1][lmax] - ht[1][lmin]), 0, 1)
g_para = {}
def getPara(radius=5):  # 根据半径计算权重参数矩阵
    global g_para
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius * 2 + 1
    m = np.zeros((size, size))
    for h in range(-radius, radius + 1):
        for w in range(-radius, radius + 1):
            if h == 0 and w == 0:
                continue
            m[radius + h, radius + w] = 1.0 / math.sqrt(h ** 2 + w ** 2)
    m /= m.sum()
    g_para[radius] = m
    return m
def zmIce(I, ratio=4, radius=5):  # 常规的ACE实现
    para = getPara(radius)
    height, width = I.shape[:2]
    zh, zw = [0] * radius +list(range(height) )+ [height - 1] * radius, [0] * radius + list(range(width)) + [width - 1] * radius
    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius * 2 + 1):
        for w in  range(radius * 2 + 1):
            if para[h][w] == 0:
                continue
            res += (para[h][w] * np.clip((I - Z[h:h + height, w:w + width]) * ratio, -1, 1))
    return res
def zmIceFast(I, ratio, radius ):  # 单通道ACE快速增强实现
    height, width = I.shape[:2]
    if min(height, width) <=2:
        return np.zeros(I.shape) + 0.5
    Rs = cv2.resize(I, (int((width + 1) /2), int((height + 1) /2)), interpolation=cv2.INTER_CUBIC)
    Rf = zmIceFast(Rs, ratio, radius )  # 递归调用
    Rf = cv2.resize(Rf, (width, height), interpolation=cv2.INTER_CUBIC)
    Rs = cv2.resize(Rs, (width, height), interpolation=cv2.INTER_CUBIC)
    return Rf + zmIce(I, ratio, radius) - zmIce(Rs, ratio, radius)
def zmIceColor(I, ratio=4, radius=10):  # rgb三通道分别增强，ratio是对比度增强因子，radius是卷积模板半径
    res = np.zeros(I.shape)
    for k in range(3):
        res[:, :, k] = stretchImage(zmIceFast(I[:, :, k], ratio, radius))
    return res
def intervalSampling(I,interval): #等间距抽样

    height, width = I.shape[:2]
    for n in range(0,interval):
        nh, nw = list(range(n, height ,interval)), list(range(width))
        smallImg= I[np.ix_(nh,nw)]
        # cv2.imwrite("data/origin_" + str(n) + "_1.png", smallImg)
        smallImg=zmIceColor(smallImg / 255.0) * 255
        I[np.ix_(nh, nw)]=smallImg
    return  I
if __name__ == '__main__':

    image=cv2.imread("data/20170626133701.jpg")
    height, width = image.shape[:2]
    image=zmIceColor(image/ 255.0) * 255
    cv2.imwrite("data/20170626133701_result.jpg",image)

    # print(sys.argv[0])
    # image=cv2.imread(sys.argv[1])
    # height, width = image.shape[:2]
    # image= cv2.resize(image, (int(sys.argv[3]), int(sys.argv[4])), interpolation=cv2.INTER_CUBIC)
    # image= cv2.resize(zmIceColor(image/ 255.0) * 255, (width, height), interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite(sys.argv[2],image)
    # print(sys.argv[2])

