# coding=utf-8
'''
cv2.arcLength: 计算轮廓的周长, True表示轮廓闭合, False表示曲线,即轮廓开放

'''
import cv2
import numpy as np 

# 0: 读入灰度图, 1: 读入彩色图
image = cv2.imread('./ell.jpg', 0)
print(image.shape)

# otsuThe, maxValue = 0, 255
# # otsuThe, dst_Otsu = cv2.threshold(image, otsuThe, maxValue, cv2.THRESH_OTSU)
# # print(otsuThe)   
# _, dst_Otsu = cv2.threshold(image, 0, 255, 0)
# kernel = np.ones((20, 20), dtype=np.uint8)
# dst_Otsu1 = cv2.erode(dst_Otsu, kernel, iterations=1)
# kernel = np.ones((5, 5), dtype=np.uint8)
# dst_Otsu2 = cv2.dilate(dst_Otsu1, kernel, 5)  

contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(contours)
print(cv2.arcLength(contours[0], True))
# cnt = contours[0]
# cnt = cnt.reshape(cnt.shape[0], -1)
# # 得到缺陷的最小外接旋转矩形
# rect = cv2.minAreaRect(cnt)
# # 得到旋转矩形的端点
# box = cv2.boxPoints(rect)
# box_d = np.int0(box)      
# image = cv2.imread('./ell.jpg')       
# cv2.drawContours(image, [box_d], 0, (0, 255, 0), 1)
# cv2.imwrite('./binary.jpg', image)
