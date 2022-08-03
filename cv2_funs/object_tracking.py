# coding:utf-8
# createBackgroundSubtractorKNN or createBackgroundSubtractorMOG2 实现目标跟踪

# code1 
# import cv2

# # 获取摄像头
# camera = cv2.VideoCapture(0)
# # 获取背景分割器对象
# bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)


# while True:
#     # 读取帧
#     ret, frame = camera.read()
#     # 获取前景
#     fgmask = bs.apply(frame)
    
#     # 自适应阈值先试试, 感受下阈值设置多少合适区分前背景
#     otsuThe, maxValue = 0, 255
#     otsuThe, dst_Otsu = cv2.threshold(fgmask.copy(), otsuThe, maxValue, cv2.THRESH_OTSU)
#     print(otsuThe)   # 127 
#     # 对前景二值化
#     th = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1]

#     # 膨胀运算
#     dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
#     # 检测轮廓
#     # cv2.imwrite('./dilated.jpg', dilated)
#     image, contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # 将轮廓画在原图像上
#     for c in contours:
#         try:
#             x, y, w, h = cv2.boundingRect(c)
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
#         except:
#             continue 
#     # 显示前景
#     # cv2.imshow("fgmask", fgmask)
#     # 显示二值化
#     cv2.imshow("thresh", th)
#     # 显示带有轮廓的原图
#     # cv2.imshow("detection", frame)
#     if cv2.waitKey(5) & 0xff == ord("q"):
#         break

# cv2.destroyAllWindows()
# camera.release()


# code2官网code
import numpy as np
import cv2

# read the video
cap = cv2.VideoCapture('./vtest.mp4')

# create the subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=100, detectShadows=False)


def getPerson(image, opt=1):

    # get the front mask
    mask = fgbg.apply(frame)

    # eliminate the noise
    line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5), (-1, -1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, line)
    cv2.imwrite('./mask.jpg', mask)
    # find the max area contours
    out, contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for count in contours:
        print(count)
        area = cv2.contourArea(count)
        if area < 150:
            continue
        rect = cv2.minAreaRect(count)
        cv2.ellipse(image, rect, (0, 255, 0), 2, 8)
        cv2.circle(image, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (255, 0, 0), 2, 8, 0)
    return image, mask


while True:
    ret, frame = cap.read()
    cv2.imwrite("input.png", frame)
    # cv2.imshow('input', frame)
    result, m_ = getPerson(frame)
    cv2.imshow('result', result)
    k = cv2.waitKey(50)&0xff
    if k == 27:
        cv2.imwrite("result.png", result)
        cv2.imwrite("mask.png", m_)

        break
cap.release()
cv2.destroyAllWindows()