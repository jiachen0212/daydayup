# coding=utf-8
import numpy as np 
import random
import cv2 


print('马赛克算法: 方块mask, 毛边mask')
def image_mask(h_range, w_range, img_name, winds=8):
	lena = cv2.imread(img_name)
	for i in range(h_range[0], h_range[1]):
		for j in range(w_range[0], w_range[1]):
			# 用8x8窗口扫描,取窗口左上角value带图这8x8内的所有像素值.
			# 可起到方块马赛克作用
			if i%winds == j%winds == 0:
				mask_value = lena[i][j]
				for a_ in range(winds):
					for b_ in range(winds):
						lena[i+a_][j+b_] = mask_value
	cv2.imwrite('./mask_lena.jpg', lena)

h_range = [90, 140]
w_range = [80, 180]
img_name = '/Users/chenjia/Desktop/lena.png'
image_mask(h_range, w_range, img_name)

def image_mask_maoboli(img_name, winds=8):
	# 让像素随机被周围像素替换, 
	lena = cv2.imread(img_name)
	# 存放mask结果图像
	dst = np.zeros(lena.shape, np.uint8)

	h, w = lena.shape[:2]
	for i in range(h):
		for j in range(w):
			# 8x8窗口内, 随机找一个像素取value,来代替[i,j]处的像素值
			off_h, off_w = random.randint(0, winds), random.randint(0, winds)
			off_h += i 
			off_w += j 
			# 图像边界越界处理:
			off_h = min(h-1, off_h)
			off_w = min(w-1, off_w)
			dst[i][j] = lena[off_h][off_w]
	cv2.imwrite('./maobian_mask_lena.jpg', dst)
image_mask_maoboli(img_name)