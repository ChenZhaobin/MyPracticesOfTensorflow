# Python3

import urllib.request
import cv2
import os
import numpy as np
# 创建图片保存目录
initDir='data/eyeglasses_detect/eyeglasses_neg'
if not os.path.exists(initDir):
    os.makedirs(initDir)

init_img_url = ['http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00523513',
               'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152']
# init_img_url=['http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03443912']

urls = ''
for img_url in init_img_url:
    urls += urllib.request.urlopen(img_url).read().decode()

img_index = 1
for url in urls.split('\n'):
    try:
        print(url)
        urllib.request.urlretrieve(url, initDir+'/' + str(img_index) + '.jpg')
        # 把图片转为灰度图片
        gray_img = cv2.imread(initDir+'/' + str(img_index) + '.jpg', cv2.IMREAD_GRAYSCALE)
        # 更改图像大小
        image = cv2.resize(gray_img, (150, 150))
        # 保存图片
        cv2.imwrite(initDir+'/' + str(img_index) + '.jpg', image)
        img_index += 1
    except Exception as e:
        print(e)


# 判断两张图片是否完全一样
def is_same_image(img_file1, img_file2):
    img1 = cv2.imread(img_file1)
    img2 = cv2.imread(img_file2)
    if img1.shape == img2.shape and not (np.bitwise_xor(img1, img2).any()):
        return True
    else:
        return False


# 去除重复图片
"""
file_list = os.listdir('neg')
try:
	for img1 in file_list:
		for img2 in file_list:
			if img1 != img2:
				if is_same_image('neg/'+img1, 'neg/'+img2) is True:
					print(img1, img2)
					os.remove('neg/'+img1)
		file_list.remove(img1)
except Exception as e:
	print(e)
"""