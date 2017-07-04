import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("data/6.jpg" ) #直接读为灰度图像
#9---滤波领域直径
#后面两个数字：空间高斯函数标准差，灰度值相似性标准差
blur = cv2.bilateralFilter(img,9,75,75)
cv2.imwrite("data/6_blur_result.jpg",blur)
cv2.imshow('blur ', blur )
cv2.waitKey()
cv2.destroyAllWindows()