import numpy as np
import cv2 
import random




def add_gauss(im):
  H, W = im.shape[:2]
  mean = 0
  sigma = 5
  gauss = np.random.normal(mean,sigma,(H,W))
  noisy_img = im + gauss
  noisy_img = np.clip(noisy_img,a_min=0,a_max=255)

  return noisy_img



def gauss_kernel(n, sigma=1):
  gauss_wind = [[0 for i in range(n)] for j in range(n)]
  for i in range(n):
    for j in range(n):
      gauss_wind[i][j] = 1/(2*np.pi*sigma**2)*np.exp(-(i**2+j**2)/(2*sigma**2)) 

  return gauss_wind


im = cv2.imread('./lena.png', 0)
im = add_gauss(im)
cv2.imwrite('./gauss_lena.jpg', im)
r = 5
H, W = im.shape[:2]
res = np.zeros((H,W))
for i in range(H-r):
  for j in range(W-r):
    im_ = im[i: i+r, j:j+r]
    res[i: i+r, j:j+r] = np.matmul(im_, gauss_kernel(r, sigma=0.5))
cv2.imwrite('./gauss_filter_lena.jpg', res)
        


