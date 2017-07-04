# -*- coding: utf-8 -*-
#线性锐化滤波-拉普拉斯算子进行二维卷积计算
#code:myhaspl@myhaspl.com
import cv2
import numpy as np
from scipy import signal
myimg=cv2.imread("data/6.jpg")
myh=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
for k in range(3):
    myimg[:, :, k] =signal.convolve2d(myimg[:, :, k] ,myh,mode="same")
cv2.imwrite("data/6_blur_sharpen_result.jpg", myimg)
cv2.imshow('src', myimg)
cv2.waitKey()
cv2.destroyAllWindows()