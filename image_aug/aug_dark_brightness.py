# 改进图像亮度, 过曝光降低亮度, 低曝光增强亮度
import numpy as np
import cv2
from math import ceil
from scipy.sparse import spdiags
from scipy.optimize import fminbound
from scipy.stats import entropy
from scipy.sparse.linalg import spsolve

def computeTextureWeights(fin, sigma, sharpness):
    # 计算纹理信息: 垂直,水平方向
    dt0_v = np.diff(fin, 1, 0)  # 垂直差分
    dt0_v = np.concatenate((dt0_v, fin[:1, :] - fin[-1:, :]), axis=0)  # 第0行减去最后一行
    dt0_h = np.diff(fin, 1, 1)  # 水平差分
    dt0_h = np.concatenate((dt0_h, fin[:, :1] - fin[:, -1:]), axis=1)  # 第0列减去最后一列
    gauker_h = cv2.filter2D(dt0_h, -1, np.ones((1, sigma)), borderType=cv2.BORDER_CONSTANT)
    gauker_v = cv2.filter2D(dt0_v, -1, np.ones((sigma, 1)), borderType=cv2.BORDER_CONSTANT)
    W_h = 1.0 / (abs(gauker_h) * abs(dt0_h) + sharpness)
    W_v = 1.0 / (abs(gauker_v) * abs(dt0_v) + sharpness)

    return W_h, W_v


def convertCol(tmp): 
    return np.reshape(tmp.T, (tmp.shape[0] * tmp.shape[1], 1))


def solveLinearEquation(IN, wx, wy, lambd):
    r, c, ch = IN.shape[0], IN.shape[1], 1
    k = r * c
    dx = -lambd * convertCol(wx)   
    dy = -lambd * convertCol(wy)
    tempx = np.concatenate((wx[:, -1:], wx[:, 0:-1]), 1)  
    tempy = np.concatenate((wy[-1:, :], wy[0:-1, :]), 0) 
    dxa = -lambd * convertCol(tempx)
    dya = -lambd * convertCol(tempy)
    tempx = np.concatenate((wx[:, -1:], np.zeros((r, c - 1))), 1)   
    tempy = np.concatenate((wy[-1:, :], np.zeros((r - 1, c))), 0)   
    dxd1 = -lambd * convertCol(tempx)
    dyd1 = -lambd * convertCol(tempy)
    wx[:, -1:] = 0   
    wy[-1:, :] = 0   
    dxd2 = -lambd * convertCol(wx)
    dyd2 = -lambd * convertCol(wy)

    # 从对角线返回一个稀疏矩阵
    Ax = spdiags(np.concatenate((dxd1, dxd2), 1).T, np.array([-k + r, -r]), k, k)
    Ay = spdiags(np.concatenate((dyd1, dyd2), 1).T, np.array([-r + 1, -1]), k, k)
    D = 1 - (dx + dy + dxa + dya)
    A = (Ax + Ay) + (Ax + Ay).T + spdiags(D.T, np.array([0]), k, k)
    A = A / 1000.0   
    matCol = convertCol(IN)
    OUT = spsolve(A, matCol, permc_spec="MMD_AT_PLUS_A")
    OUT = OUT / 1000
    OUT = np.reshape(OUT, (c, r)).T

    return OUT


def tsmooth(I, lambd=0.5, sigma=5, sharpness=0.001):
    # 计算纹理
    # 解线性方程
    wx, wy = computeTextureWeights(I, sigma, sharpness)
    S = solveLinearEquation(I, wx, wy, lambd)
    return S


def rgb2gm(I):
    if I.shape[2] and I.shape[2] == 3:
        I = np.power(np.multiply(np.multiply(I[:, :, 0], I[:, :, 1]), I[:, :, 2]), (1.0 / 3))
    return I


def applyK(I, k, a=-0.3293, b=1.1258):
    beta = np.exp((1 - (k ** a)) * b)
    gamma = (k ** a)
    BTF = np.power(I, gamma) * beta
    
    return BTF


def maxEntropyEnhance(I, isBad, mink=1, maxk=10):
    Y = I / 255.0
    Y = rgb2gm(Y)
    Y = Y[isBad >= 1]

    def f(k):
        return -entropy(applyK(Y, k))
    opt_k = fminbound(f, mink, maxk)
    J = applyK(I, opt_k) - 0.01
    return J


def oneHDR(image_path,  mu=0.5, a=-0.3293, b=1.1258):
    I = cv2.imread(image_path)
    # mu: 图像增强程度, 越大增强指数越高
    I = I / 255.0
    t_b = I[:, :, 0] 
    for i in range(I.shape[2]-1): 
        t_b = np.maximum(t_b, I[:, :, i + 1])
 
    # 求照度图T 
    t_our = tsmooth(t_b) 
    t = cv2.merge([t_our, t_our, t_our])
    W = t ** mu  
    # 可看看照度函数的逆, 长啥样 
    cv2.imwrite(filepath + '1-t.jpg', (1 - t) * 255)

    # 变暗
    I2 = I * W  # 原图*权重
    # 曝光率->k ->J
    isBad = t_our < 0.5   
    # 最大熵增强 
    J = maxEntropyEnhance(I, isBad)  # 求k和曝光图
    J2 = J * (1 - W)  # 曝光图*权重
    fused = I2 + J2  # 增强图
    
    return fused


if __name__ == '__main__':
    
    auged_image = oneHDR(image_path=None)