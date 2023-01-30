import cv2
import numpy as np
import time

def slow_MeanFilter(im,r):
    H,W = im.shape
    res = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            s,n=0,0
            for k in range(i-r//2,i+r-r//2):
                for m in range(j-r//2,j+r-r//2):
                    if k<0 or k>=H or m<0 or m>=W:
                        continue
                    else:
                        s += im[k,m]
                        n += 1
            res[i,j] = s/n
    return res



def MeanFilter(im, r):
    '''
    im灰度图, r为滤波半径

    '''

    H,W = im.shape 
    res = np.zeros((H,W))
    
    # 行维度上, 由上至下逐渐加法, 做行上的积分
    integralImagex = np.zeros((H+1,W))
    for i in range(H):    
        integralImagex[i+1,:] = integralImagex[i,:]+im[i,:]
    # 以r为单位, 对应做减法, 起到下面多一行, 上面就减一行的效果 
    # 得到的mid, 就是r行上, 各行的积分值(完成了行上加法)
    mid = integralImagex[r:]-integralImagex[:-r]
    # /r 就是在行维度上做mean处理   
    mid = mid / r  

    # 行上做左右padding
    padding = r - 1 
    leftPadding = (r-1)//2 
    rightPadding = padding - leftPadding

    # 基本后第i行值padding
    left = integralImagex[r-leftPadding:r]
    # 原im的值padding
    right = integralImagex[-1:] - integralImagex[-r+1:-r+1+rightPadding]

    leftNorm = np.array(range(r-leftPadding,r,1)).reshape(-1,1)
    rightNorm = np.array(range(r-1,r-rightPadding-1,-1)).reshape(-1,1)
    left /= leftNorm
    right /= rightNorm
    im1 = np.concatenate((left,mid,right))

    # 相同方式处理列
    integralImagey = np.zeros((H,W+1))
    res = np.zeros((H,W))
    for i in range(W):    
        integralImagey[:,i+1] = integralImagey[:,i]+im1[:,i]
    mid = integralImagey[:,r:]-integralImagey[:,:-r] 
    mid = mid / r 
    left = integralImagey[:,r-leftPadding:r]
    right = integralImagey[:,-1:] - integralImagey[:,-r+1:-r+1+rightPadding]
    leftNorm = np.array(range(r-leftPadding,r,1)).reshape(1,-1)
    rightNorm = np.array(range(r-1,r-rightPadding-1,-1)).reshape(1,-1)
    left /= leftNorm
    right /= rightNorm
    im2 = np.concatenate((left,mid,right),axis=1)
    
    return im2

im = cv2.imread('/Users/chenjia/Downloads/Smartmore/2022/daydayup/图像马赛克_方块_毛边/lena.png', 0)
r = 3
start = time.time()
res = MeanFilter(im,r)
print('quick: ', time.time()-start)

start1 = time.time()
res = slow_MeanFilter(im, r)
print('slow: ', time.time()-start1)
cv2.imwrite('./filter.jpg', res)