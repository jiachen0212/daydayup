# coding=utf-8
import matplotlib.pyplot as plt
import cv2 

#1. blur合集
img = cv2.imread("./1.jpg")
plt.subplot(231) 
plt.xlabel('img')
plt.imshow(img[..., ::-1])
mean_blur = cv2.blur(img,(11,11))
plt.subplot(232)
plt.xlabel('mean_blur')
plt.imshow(mean_blur[..., ::-1])
Gauss_blur = cv2.GaussianBlur(img,(11,11),0)
plt.subplot(233)
plt.xlabel('Gauss_blur')
plt.imshow(Gauss_blur[..., ::-1])
# 双边滤波
Bilateral_blur = cv2.bilateralFilter(img,11,75,75)
plt.subplot(234)
plt.xlabel('Bilateral_blur')
plt.imshow(Bilateral_blur[..., ::-1])
MedianBlur = cv2.medianBlur(img, 11)
plt.subplot(235)
plt.xlabel('MedianBlur')
plt.imshow(MedianBlur[..., ::-1])
plt.show()


#2. gamma 直方图均衡
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
 
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)
 
img = cv2.imread('./test.jpg', 1)
plt.subplot(131)
plt.xlabel('org')
plt.imshow(img[..., ::-1])
# 直方图均衡
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH))
plt.subplot(132)
plt.xlabel('equalizeHist')
plt.imshow(result[..., ::-1])
# gamma变换, 使用了cv2.LUT()快速查边函数
img_gamma = adjust_gamma(img, gamma=2)
plt.subplot(133)
plt.xlabel('gamma_aug')
plt.imshow(img_gamma[..., ::-1])
plt.show()
