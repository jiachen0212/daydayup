import cv2
import random
import numpy as np 

src = cv2.imread(cv2.samples.findFile('./cat.jpeg'))
gray = cv2.cvtColor(src,cv2.COLOR_BGRA2GRAY)
cv2.imwrite("gray.jpg", src)

# 存储边缘分割结果
ssrc = src.copy()*0
# 存储线检测结果
lsrc = src.copy()
# 存储椭圆检测结果
esrc = src.copy()

# opencv的边缘提取方法, 不同于canny等减法边缘策略.
# https://docs.opencv.org/3.4/d4/d8b/group__ximgproc__edge__drawing.html 官方解说文档

ed = cv2.ximgproc.createEdgeDrawing()

# you can change parameters (refer the documentation to see all parameters) 设置参数
EDParams = cv2.ximgproc_EdgeDrawing_Params()
EDParams.MinPathLength = 50    # try changing this value between 5 to 1000
EDParams.PFmode = True      # false画椭圆, True画圆 
EDParams.MinLineLength = 10    # try changing this value between 5 to 100
EDParams.NFAValidation = True   # defaut value try to swich it to False
ed.setParams(EDParams)

# Detect edges
# you should call this before detectLines() and detectEllipses()
ed.detectEdges(gray)
segments = ed.getSegments()
lines = ed.detectLines()
ellipses = ed.detectEllipses()

# Draw detected edge segments
for i in range(len(segments)):
    color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
    cv2.polylines(ssrc, [segments[i]], False, color, 1, cv2.LINE_8)

cv2.imwrite("edge_segments.jpg", ssrc)

# Draw detected lines
if lines is not None: # Check if the lines have been found and only then iterate over these and add them to the image
    lines = np.uint16(np.around(lines))
    for i in range(len(lines)):
        cv2.line(lsrc, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 1, cv2.LINE_AA)

cv2.imwrite("lines.jpg", lsrc)

# Draw detected circles and ellipses
if ellipses is not None: # Check if circles and ellipses have been found and only then iterate over these and add them to the image
    ellipses = np.uint16(np.around(ellipses))
    for i in range(len(ellipses)):
        color = (0, 0, 255)
        if ellipses[i][0][2] == 0:
            color = (0, 255, 0)
        cv2.ellipse(esrc, (ellipses[i][0][0], ellipses[i][0][1]), (ellipses[i][0][2]+ellipses[i][0][3],ellipses[i][0][2]+ellipses[i][0][4]),ellipses[i][0][5],0, 360, color, 2, cv2.LINE_AA)

cv2.imwrite("ellipses.jpg", esrc)
