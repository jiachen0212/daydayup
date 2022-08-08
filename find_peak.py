# coding=utf-8

'''
寻找序列数据的波峰(峰值)
'''

import matplotlib.pyplot as plt
import numpy as np


def AMPD(data):
    """
    实现AMPD算法
    :param data: 1-D numpy.ndarray 
    :return: 波峰所在索引值的列表
    """
    p_data = np.zeros_like(data, dtype=np.int32)
    count = data.shape[0]
    arr_rowsum = []
    for k in range(1, count // 2 + 1):
        row_sum = 0
        for i in range(k, count - k):
            # data[i] > data[i - k]: 对称的右边大于左边
            # data[i] > data[i + k]: 当前点大于右边的值, 也是它是突出的值
            if data[i] > data[i - k] and data[i] > data[i + k]:
                row_sum -= 1
        arr_rowsum.append(row_sum)
    # 找到在多少周期内可以画作一个window, 在此内找peak值
    min_index = np.argmin(arr_rowsum)
    max_window_length = min_index
    for k in range(1, max_window_length + 1):
        for i in range(k, count - k):
            if data[i] > data[i - k] and data[i] > data[i + k]:
                p_data[i] += 1
    return np.where(p_data == max_window_length)[0]


def sim_data():
    N = 1000
    x = np.linspace(0, 200, N)
    y = 2 * np.cos(2 * np.pi * 300 * x) + 5 * np.sin(2 * np.pi * 100 * x) + 4 * np.random.randn(N)

    return y

def main():
    # 构造序列数据
    y = sim_data()
    plt.plot(range(len(y)), y)
    px = AMPD(y)
    # 画peak点
    plt.scatter(px, y[px], color="red")

    plt.show()

main()



    
