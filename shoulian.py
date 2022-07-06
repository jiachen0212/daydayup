# coding=utf-8

# 瘦脸算法的原理:
    # shader(纹理) 对像素位置进行偏移来实现对脸部区域的放大缩小：根据特定的坐标映射关系,对应原始和变换后的坐标.
    # 根据目的坐标, 算法原始坐标. 使用插值等手法, 或者原始坐标处的像素值, 然后赋值给目的坐标.


# https://blog.csdn.net/qq_14845119/article/details/121500720

import cv2
import math
import numpy as np
import imageio
 
def localTranslationWarpFastWithStrength(srcImg, startX, startY, endX, endY, radius, strength):
    ddradius = float(radius * radius)
    copyImg = np.zeros(srcImg.shape, np.uint8)
    copyImg = srcImg.copy()
 
 
    maskImg = np.zeros(srcImg.shape[:2], np.uint8)
    cv2.circle(maskImg, (startX, startY), math.ceil(radius), (255, 255, 255), -1)
 
    K0 = 100/strength
 
    # 计算公式中的|m-c|^2
    ddmc_x = (endX - startX) * (endX - startX)
    ddmc_y = (endY - startY) * (endY - startY)
    H, W, C = srcImg.shape
 
    mapX = np.vstack([np.arange(W).astype(np.float32).reshape(1, -1)] * H)
    mapY = np.hstack([np.arange(H).astype(np.float32).reshape(-1, 1)] * W)
 
    distance_x = (mapX - startX) * (mapX - startX)
    distance_y = (mapY - startY) * (mapY - startY)
    distance = distance_x + distance_y
    K1 = np.sqrt(distance)
    ratio_x = (ddradius - distance_x) / (ddradius - distance_x + K0 * ddmc_x)
    ratio_y = (ddradius - distance_y) / (ddradius - distance_y + K0 * ddmc_y)
    ratio_x = ratio_x * ratio_x
    ratio_y = ratio_y * ratio_y
 
    UX = mapX - ratio_x * (endX - startX) * (1 - K1/radius)
    UY = mapY - ratio_y * (endY - startY) * (1 - K1/radius)
 
    np.copyto(UX, mapX, where=maskImg == 0)
    np.copyto(UY, mapY, where=maskImg == 0)
    UX = UX.astype(np.float32)
    UY = UY.astype(np.float32)
    copyImg = cv2.remap(srcImg, UX, UY, interpolation=cv2.INTER_LINEAR)
 
    return copyImg
 
 
image = cv2.imread("./yetai_saunfa.PNG")
image = image[:, :, ::-1]  # BGR2RGB
processed_image = image.copy()
startX_left, startY_left, endX_left, endY_left = 101, 266, 192, 233
startX_right, startY_right, endX_right, endY_right = 287, 275, 192, 233
radius = 45
strength = 100
# 瘦左边脸                                                                           
processed_image = localTranslationWarpFastWithStrength(processed_image, startX_left, startY_left, endX_left, endY_left, radius, strength)
# 瘦右边脸                                                                           
processed_image = localTranslationWarpFastWithStrength(processed_image, startX_right, startY_right, endX_right, endY_right, radius, strength)
cv2.imwrite("./thin.jpg", processed_image[:, :, ::-1])

# 把瘦脸前后的图合并成gif
imageio.mimwrite('./shoulian.gif', [image, processed_image], duration=0.35)


# startX_left, startY_left, endX_left, endY_left, startX_right, startY_right, endX_right, endY_right
point_size = 1
point_color = [(0, 0, 255),(0, 255, 0),(255, 0, 0),(128, 0, 128)]  # bgr
thickness = 4  
 
coordinates = [[startX_left, startY_left], [endX_left, endY_left], [startX_right, startY_right], [endX_right, endY_right]]
imgs = [image.copy(), image.copy(), image.copy(), image.copy()]
for ind, coor in enumerate(coordinates):
    print(coor)
    cv2.circle(imgs[ind], (int(coor[0]),int(coor[1])), point_size, point_color[ind], thickness)
imageio.mimwrite('./points.gif', imgs, duration=0.35)
 
