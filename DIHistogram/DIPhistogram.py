#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 20:40:01 2022

@author: anhvietpham

@Document
https://www.geeksforgeeks.org/opencv-python-program-analyze-image-using-histogram/
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12,10]

iname = "/Users/anhvietpham/Downloads/INT3404_20-master/week3/code/flowers/3.jpg"

if type(iname) == str:
    #read image
    img1 = cv2.imread(iname)
else:
    img1 = iname
    
if len(img1.shape) == 3:
    #convert to gray
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
else:
    gray1 = iname

resized_image = cv2.resize(gray1, (int(gray1.shape[1] / 2,), int(gray1.shape[0] / 2)), interpolation = cv2.INTER_AREA)


#size of image
h1, w1 = img1.shape[:2]

h2, w2 = resized_image.shape[:2]


#calculate histogram
hist1 = cv2.calcHist(
                    [gray1],
                    channels=[0],
                    mask=None, #full image
                    histSize=[256], #full scale
                    ranges=[0,256]
)

hist2 = cv2.calcHist(
                    [resized_image],
                    channels=[0],
                    mask=None, #full image
                    histSize=[256], #full scale
                    ranges=[0,256]
)

# normalized histogram
norm_hist1 = hist1/(h1*w1)
#cumulative histogram
cdf1 = norm_hist1.cumsum()

# normalized histogram
norm_hist2 = hist2/(h2*w2)
#cumulative histogram
cdf2 = norm_hist2.cumsum()
    
img_show = img1[:,:,::-1]
    
#plot image
plt.subplot(2,2,1)
plt.imshow(img1[:,:,::-1])
plt.title("color image")
plt.subplot(2, 2, 2)
plt.stem(range(256), hist1, use_line_collection=True)
plt.title("histogram")
plt.subplot(2, 2, 3)
plt.imshow(gray1, cmap='gray')
plt.title("grayscale image")
plt.subplot(2, 2, 4)
plt.step(range(256), cdf1, c='g')
plt.title("cumulative histogram")
plt.show()

    
