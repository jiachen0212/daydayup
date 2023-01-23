# coding=utf-8
# 去水印算法
from os.path import splitext
# 计算笛卡尔积: 两个集合的所有组合,
from itertools import product
from PIL import Image
 


def qu_shuiyin(fn, pixel_thres=580):
	im = Image.open(fn)
	width, height = im.size
	for pos in product(range(width), range(height)):
	    # 这个pos其实就是所有图像点坐标
	    # 580是经验值, RGB值之和>580则认为此处是水印. 
	    if sum(im.getpixel(pos)[:3]) > pixel_thres:
	        im.putpixel(pos, (250,250,250))  # 重新pixel赋值 
	im.save('_无水印'.join((splitext(fn))))

fn = './sy.png'
qu_shuiyin(fn)