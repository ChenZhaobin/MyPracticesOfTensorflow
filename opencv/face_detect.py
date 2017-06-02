import cv2
img = cv2.imread("data/test.jpg")
# 加载分类器
face_haar = cv2.CascadeClassifier("data/face_detect/haarcascade_frontalface_default.xml")
eye_haar = cv2.CascadeClassifier("data/face_detect/haarcascade_eye.xml")
# 把图像转为黑白图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 检测图像中的所有脸
faces = face_haar.detectMultiScale(gray_img, 1.3, 5)
for face_x, face_y, face_w, face_h in faces:
    cv2.rectangle(img, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)
    # 眼长在脸上
    roi_gray_img = gray_img[face_y:face_y + face_h, face_x:face_x + face_w]
    roi_img = img[face_y:face_y + face_h, face_x:face_x + face_w]
    eyes = eye_haar.detectMultiScale(roi_gray_img, 1.3, 5)
    for eye_x, eye_y, eye_w, eye_h in eyes:
        cv2.rectangle(roi_img, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255, 0, 0), 2)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()