import cv2
import numpy as np
import os

def simpleEllipse2Circle(img, a, b, angle):
    """
    args:
       img: 包含椭圆的图片
       a: 椭圆短直径
       b: 椭圆长直径
       angle: 旋转角度
    returns:
        包含椭圆变成圆形的图片
    """
    h, w = img.shape[:2]
    scale = a / b
    
    ##得到仿射矩阵 2*3   2*2旋转 + 2*1平移
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    M[:,0:2] = np.array([[1,0],[0, scale]]) @ M[:,0:2]
   
    M[1,2] = M[1,2] * scale 
    transform = M[:, 0:2]
    ##生成的圆的中心坐标
    #cneter_pt = transform @ np.array([e_x,e_y]) + M[:,2]
    # 生成圆的半径等于原椭圆短直径/2
    #r = a / 2
    ## top_pt = [ptret[0], ptret[1]-r] 
    circle_out = cv2.warpAffine(img, M, (w, h), borderValue=255)
    #mask_img = np.zeros_like(circle_out)
    #cv2.circle(mask_img, (int(cneter_pt[0]), int(cneter_pt[1])), int(r), 255, -1)
    #circle_out[mask_img!=255] = 255
    return circle_out

if __name__ == '__main__':

    img_path = "./ell.jpg"
    img = cv2.imread(img_path, 0)
    # print(img.shape)
    # 先自适应分割得到参考阈值, 再微调分割出椭圆
    otsuThe, maxValue = 0, 255
    otsuThe, dst_Otsu = cv2.threshold(img, otsuThe, maxValue, cv2.THRESH_OTSU)
    # print(otsuThe)  
    # _, thresh = cv2.threshold(img, 126, 255, type=cv2.THRESH_BINARY_INV)
    # cv2.imshow('1', otsuThe)
    # cv2.waitKey(10000)
    # cv2.imwrite('./gray.jpg', dst_Otsu)

    contours, _ = cv2.findContours(dst_Otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 寻找到多个轮廓点集的话, 使用点数最多的那个集合
    count_lens = [len(a) for a in contours]
    count = contours[count_lens.index(max(count_lens))]
    # cv2.fitEllipse: 获取椭圆的长短半径, 俩圆心等参数. 输入为椭圆的外围有序点集 
    params = cv2.fitEllipse(count)

    e_x, e_y = params[0]
    a, b = params[1]
    angle = params[2]
    print(params)
    
    import time
    s = time.time()
    circle_out = simpleEllipse2Circle(img, a, b, angle)
    e = time.time()
    print('caost time:', e-s)
    cv2.imwrite('circular.jpg', circle_out)