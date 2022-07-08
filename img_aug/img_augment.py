# coding=utf-8
import cv2
import numpy as np
import random


def adjust_brightness(img_path, brightness_factor):
    img = cv2.imread(img_path)
    # clip(0, 255)会把处理后的像素值的大小，现在在[0, 255]范围内，如果有值大于255则取255,如果有值小于0则取值0
    table = np.array([i * brightness_factor for i in range (0,256)]).clip(0,255).astype('uint8')
    # 单通道img
    if img.shape[2] == 1:
        return cv2.LUT(img, table)[:,:,np.newaxis]
    # 多通道img
    else:
        # cv2.LUT()查表替换像素值
        result = cv2.LUT(img, table)
        # 左边原图、右边增加亮度后的图
        imgs_hstack = np.hstack((img, result))
        # cv2.imshow("result", imgs_hstack)
        # cv2.waitKey(0)
        return imgs_hstack   # result
 


def cv2_letterbox_image(image, expected_size):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    # 上下左右需要填充的像素点个数
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)

    return new_img



def roata_and_scale(img, degrees=5, scale_r=1.2):
    rows, cols = img.shape[:2] 
    new_rows, new_cols = int(rows*scale_r), int(cols*scale_r)
    # 旋转中心, 旋转角度, 图像缩放比.  
    center = (random.randint(rows//4, rows*3//4), random.randint(cols//4, cols*3//4))
    # 得到旋转矩阵M 
    M = cv2.getRotationMatrix2D(center, degrees, scale_r) 
    #第三个参数：变换后的图像大小 
    res = cv2.warpAffine(img,M,(new_cols, new_rows)) 

    return res 



if __name__ == '__main__':


    # 1. cv2.LUT()查表替换像素值, 增强暗图像的亮度.
    # img_path, brightness_factor = './dark.PNG', 2
    # res = adjust_brightness(img_path, brightness_factor)
    # cv2.imwrite('./lut_aug_light.jpg', res)


    # 2. 目标检测中常见的data_aug. 
    # 保持原image的长宽比,且resize到目标size.(上下左右用黑色填充, 使图像不变形且是目标size大小.)
    # image = cv2.imread('./cat.PNG')
    # expected_size = (1300, 1300)
    # res = cv2_letterbox_image(image, expected_size)
    # cv2.imwrite('./letterbox_cat.jpg', res)

    
    # 3. 实现任意点旋转矩形并修改图像尺寸
    # cv2.getRotationMatrix2D and  cv2.warpAffine
    image = cv2.imread('./cat.PNG')
    roata_scale_res = roata_and_scale(image, degrees=5, scale_r=1.2)
    cv2.imwrite('./rotat_rescale_cat.jpg', roata_scale_res)

