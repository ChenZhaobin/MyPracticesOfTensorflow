import cv2
import time
for  k in range(16,30):
    print(k)
    time1=time.time()
    img = cv2.imread("data/"+str(k)+".jpg")
    # 加载分类器
    face_haar = cv2.CascadeClassifier("data/spermclassifier/cascade.xml")
    # 检测图像中的所有脸
    faces = face_haar.detectMultiScale(img , 1.3, 5)
    for face_x, face_y, face_w, face_h in faces:
        cv2.rectangle(img, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)
        # 眼长在脸上
        roi_gray_img = img [face_y:face_y + face_h, face_x:face_x + face_w]
        roi_img = img[face_y:face_y + face_h, face_x:face_x + face_w]
    time2=time.time();
    print(time2-time1)
    cv2.imwrite("data/"+str(k)+"_result3.jpg",img)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()