# coding=utf-8
# 多边形填充: cv2.fillConvexPoly && cv2.fillPoly()
# 可以是四边五边等..
import numpy as np 
import cv2


img_gray = cv2.imread('./cat.jpeg', 0)
H, W = img_gray.shape[:2]
# 1.遮住上1/4图像
rectangular = np.array([[0,0], [0,H//4], [W,H//4], [W,0]])
cv2.fillConvexPoly(img_gray, rectangular, (0,0,0))
cv2.imwrite('./hidden_cat.jpg', img_gray)

# 2. cv2.fillPoly(img_gray, [rec1, rec2, rec3, rec4], (255,255,255))
rec1 = np.array([[0,0], [0,385], [W,385], [W,0]])
rec2 = np.array([[0,385], [0,740], [740,740], [740,385]])
rec3 = np.array([[0,740], [0,H], [W,H], [W,740]])
rec4 = np.array([[1030,385], [1030,740], [W,740], [W,385]])
cv2.fillPoly(img_gray, [rec1, rec2, rec3, rec4], (255,255,255))
cv2.imwrite('./caffe.jpg', img_gray)


# 计算图像四个通道上的平均值
image = cv2.imread('/Users/chenjia/Desktop/1.png')
print(cv2.mean(image))