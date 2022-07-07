# coding=utf-8

# https://zhuanlan.zhihu.com/p/538049366

import cv2
from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt
import numpy as np



# 1. 磨皮: 滤波+锐化
img = cv2.imread("./test.jpg")
plt.subplot(221) 
plt.xlabel('img')
plt.imshow(img[..., ::-1])
mean_blur = cv2.blur(img,(7,7))
plt.subplot(222)
plt.xlabel('mean_blur')
plt.imshow(mean_blur[..., ::-1])
Gauss_blur = cv2.GaussianBlur(img,(7,7),0)
plt.subplot(223)
plt.xlabel('Gauss_blur')
plt.imshow(Gauss_blur[..., ::-1])
# 双边滤波
Bilateral_blur = cv2.bilateralFilter(img,9,75,75)
plt.subplot(224)
plt.xlabel('Bilateral_blur')
plt.imshow(Bilateral_blur[..., ::-1])
plt.show()


alpha = 0.3
beta = 1 - alpha
gamma = 0
img_add = cv2.addWeighted(img, alpha, mean_blur, beta, gamma)

# 锐度增强
img_add = Image.fromarray(img_add)
enh_sha = ImageEnhance.Sharpness(img_add)
sharpness = 1.5
image_sharped = enh_sha.enhance(sharpness)

# 对比度增强
enh_con = ImageEnhance.Contrast(image_sharped)
contrast = 1.15
image_contrasted = enh_con.enhance(contrast)
plt.imshow(np.asarray(image_contrasted)[..., ::-1])
plt.show()


# 2. 美白: RGB or HSV 空间中处理
# 1. 创建白图层+ImageEnhance包的对比度亮度增强, 美白图像.
img = cv2.imread("./test.jpg")
plt.subplot(131) 
plt.xlabel('img')
plt.imshow(img[..., ::-1])

height,width,n = img.shape
# 锐度增强
# ImageEnhance.Sharpness(img)

# 创建一个纯白的图层
# img2 = img.copy()
# for i in range(height):
#     for j in range(width):
#         img2[i, j][0] = 255
#         img2[i, j][1] = 255
#         img2[i, j][2] = 255
img2 = np.ones_like(img)*255

# img = cv2.bilateralFilter(img, 9, 75, 75)
dst=cv2.addWeighted(img, 0.6, img2, 0.4, 0)
# cv2.imwrite('meibai/res.jpg', dst)
img3 = Image.fromarray(dst)
# 对比度增强
enh_con = ImageEnhance.Contrast(img3)
contrast = 1.2
image_contrasted = enh_con.enhance(contrast)
# image_contrasted.show()
plt.subplot(132) 
plt.xlabel('imgadd255')
plt.imshow(np.array(image_contrasted)[..., ::-1])


# 亮度增强
enh_bri = ImageEnhance.Brightness(image_contrasted)
brightness = 1.1
image_brightened = enh_bri.enhance(brightness)
# image_brightened.show()
plt.subplot(133)
plt.xlabel('meibai_res')
image_brightened = np.array(image_brightened)
plt.imshow(image_brightened[..., ::-1])
plt.savefig('whiten.jpg')
plt.show()


# 2. beta参数调节图像亮度
# G(x,y)=log(f(x,y)×(beta-1)+1)/log(beta). f(x,y)是原始像素点, G(x,y)是输出像素点.
# img2[i, j][0] = alpha * math.log(img[i, j][0] * (beta - 1) + 1) / math.log(beta)
# img2[i, j][1] = alpha * math.log(img[i, j][1] * (beta - 1) + 1) / math.log(beta)
# img2[i, j][2] = alpha * math.log(img[i, j][2] * (beta - 1) + 1) / math.log(beta)


# 尝试hsv空间做美白. 效果不是很好..
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_h, img_s, img_v = cv2.normalize(hsv_img[:,:,0],None,0,255,cv2.NORM_MINMAX), cv2.normalize(hsv_img[:,:,1],None,0,255,cv2.NORM_MINMAX), cv2.normalize(hsv_img[:,:,2],None,0,255,cv2.NORM_MINMAX) 
# cv2.imwrite('./h.jpg', img_h)
# cv2.imwrite('./s.jpg', img_s)
# cv2.imwrite('./v.jpg', img_v)
# plt.subplot(131)
# plt.xlabel('h')
# plt.imshow(img_h[..., ::-1])
# plt.subplot(132)
# plt.xlabel('s')
# plt.imshow(img_s[..., ::-1])
# plt.subplot(133)
# plt.xlabel('v')
# plt.imshow(img_v[..., ::-1])
# plt.show()

# enh_v = img_v*2
# print(hsv_img[:,:,0].dtype)
enh_v = hsv_img[:,:,2]*1.4
# enh_v = enh_v.astype(np.uint8)
# print(enh_v.dtype)

enh_v_ = np.ones_like(enh_v)
for i in range(enh_v.shape[0]):
    for j in range(enh_v.shape[1]):
        # 要做这个255截断,不然会出现图像黑块.
        enh_v_[i][j] = min(255, enh_v[i][j])
enh_v_ = enh_v_.astype(np.uint8)

# enh_h = hsv_img[:,:,0]*1.3
# enh_h = enh_v.astype(np.uint8)
# enh_s = hsv_img[:,:,1]*1.3
# enh_s = enh_v.astype(np.uint8)

res = cv2.merge([hsv_img[:,:,0], hsv_img[:,:,1], enh_v_])
res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
# res = cv2.addWeighted(res, 0.6, np.ones_like(res)*255, 0.4, 0)
plt.imshow(res[..., ::-1])
plt.show()


# 2. 查表法美白
# 获取滤镜颜色表, 根据原像素的rgb值, 计算其对应滤镜表中的坐标, 然后取色.
def getBGR(img, table, i, j):
    b, g, r = img[i][j]
    # 计算标准颜色表中颜色的位置坐标
    x = int(g/4 + int(b/32) * 64)
    y = int(r/4 + int((b%32) / 4) * 64)
    # 返回滤镜颜色表中对应的颜色
    return table[x][y]

#简单使用代码
# img = cv2.imread('input.png')
# lut_map = cv2.imread('lvjing_map.png')
# rows, cols = img.shape[:2]
# dst = np.zeros((rows, cols, 3), dtype="uint8")
# for i in range(rows):
#     for j in range(cols):
#         dst[i][j] = getBGR(img, lut_map, i, j)


# 查表法的另一种实现
Color_list = [
	1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31, 33, 35, 37, 39,
	41, 43, 44, 46, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 66, 67, 69, 71, 73, 74,
	76, 78, 79, 81, 83, 84, 86, 87, 89, 91, 92, 94, 95, 97, 99, 100, 102, 103, 105,
	106, 108, 109, 111, 112, 114, 115, 117, 118, 120, 121, 123, 124, 126, 127, 128,
	130, 131, 133, 134, 135, 137, 138, 139, 141, 142, 143, 145, 146, 147, 149, 150,
	151, 153, 154, 155, 156, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 170,
	171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
	188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
	204, 205, 205, 206, 207, 208, 209, 210, 211, 211, 212, 213, 214, 215, 215, 216,
	217, 218, 219, 219, 220, 221, 222, 222, 223, 224, 224, 225, 226, 226, 227, 228,
	228, 229, 230, 230, 231, 232, 232, 233, 233, 234, 235, 235, 236, 236, 237, 237,
	238, 238, 239, 239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 244, 245,
	245, 246, 246, 246, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 250,
	251, 251, 251, 251, 252, 252, 252, 252, 253, 253, 253, 253, 253, 254, 254, 254,
	254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 256]

img = cv2.imread("./test.jpg")
img1 = cv2.bilateralFilter(img, 9, 75, 75)
height,width,n = img1.shape
img2 = img1.copy()
for i in range(height):
    for j in range(width):

        B=img2[i, j][0]
        G=img2[i, j][1]
        R=img2[i, j][2]

        img2[i, j][0] = Color_list[B]
        img2[i, j][1] = Color_list[G]
        img2[i, j][2] = Color_list[R]
       
image= Image.fromarray(img2)
# 色度增强
enh_con = ImageEnhance.Color(image)
contrast = 1.2
image_contrasted = enh_con.enhance(contrast)
plt.imshow(np.asarray(image_contrasted)[..., ::-1])
plt.show()

