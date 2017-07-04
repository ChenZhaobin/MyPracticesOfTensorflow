import cv2
import time
import sys
import numpy as np
import os
def thresholImage(originDir ,dstDir,imgCount):
    for k in range(0,imgCount):
        originSrc=originDir+"\\"+str(k)+".jpg"
        if os.path.exists(originSrc):
            dstSrc=dstDir+"\\"+str(k)+".jpg"
            img = cv2.imread(originSrc, 0)
            img = cv2.medianBlur(img, 5)
            kernel = np.uint8(np.zeros((5, 5)))
            for x in range(5):
                kernel[x, 2] = 1;
                kernel[2, x] = 1;
            img=cv2.erode(img,kernel)
            img = cv2.erode(img, kernel)
            img = cv2.dilate(img,kernel)
            img = cv2.dilate(img, kernel)
            th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                        cv2.THRESH_BINARY_INV , 11,4)
            cv2.imwrite(dstSrc, th2)
    return

if __name__ == '__main__':
        time1=time.time()
        thresholImage(sys.argv[1], sys.argv[2],int(sys.argv[3]))
        time2=time.time()
        print("用时："+str(time2-time1))