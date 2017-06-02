import os
import  subprocess
import cv2
with open('neg.txt', 'w') as f:
    for img in os.listdir('data/eyeglasses_detect/eyeglasses_neg'):
        line = 'data/eyeglasses_detect/eyeglasses_neg/' + img + '\n'
        f.write(line)
# 把图片转为灰度图片
gray_img = cv2.imread('eyeglasses.jpg', cv2.IMREAD_GRAYSCALE)
# 更改图像大小
image = cv2.resize(gray_img, (150, 150))
# 保存图片
cv2.imwrite('myglass.jpg', image)
p = subprocess.run(["opencv_createsamples", "-img", "myglass.jpg","-bg","neg.txt","-info","pos.txt","-maxxangle","0.5","-maxzangle","-0.5","-num","10"], subprocess.PIPE)
# print(p)
p = subprocess.run(["opencv_createsamples","-info","pos.txt", "-info","pos.txt","-num","10","-w","25","-h","25","-vec","pos.vec"], subprocess.PIPE)
# print(p)
p=subprocess.run(["opencv_traincascade", "-data","data/eyeglasses_detect/haar","-vec","pos.vec","-bg","neg.txt","-numPos","10","-numNeg","10","-numStages","15","-w","25","-h","25"],subprocess.PIPE)
print(p)
galsses_haar = cv2.CascadeClassifier("data/eyeglasses_detect/haar/cascade.xml")
img = cv2.imread("data/test.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# http://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
galsses = galsses_haar.detectMultiScale(gray_img, 1.2, 3)  # 调整参数
for galsses_x, galsses_y, galsses_w, galsses_h in galsses:
    cv2.rectangle(img, (galsses_x, galsses_y), (galsses_x + galsses_w, galsses_y + galsses_h), (0, 255, 0), 2)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()