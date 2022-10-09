#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 19:14:06 2022

@author: anhvietpham
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 8]

image = cv2.imread("/Users/anhvietpham/Documents/cs/Digital-Image-Processing/test-patten-image.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray_image, cmap='gray')
plt.axis("off")
plt.show()

hist1 = cv2.calcHist(
    [gray_image],
    channels=[0],
    mask=None,  # full image
    histSize=[256],  # full scale
    ranges=[0, 256]
)

counts_hist_1, bins_hist_1 = np.histogram(gray_image, bins=256, range=(0, 256))

plt.hist(counts_hist_1, bins=256, range=(0, 256))

plt.show()

n = 25

ones = np.ones((n, n))

kernel = 1 / (n * n) * ones

fimage = cv2.filter2D(src=gray_image, ddepth=-1, kernel=kernel)
plt.subplot(1, 2, 1);
plt.imshow(gray_image, cmap='gray')
plt.axis("off")
plt.subplot(1, 2, 2);
plt.imshow(fimage, cmap='gray')
plt.axis("off")
plt.show()

hist2 = cv2.calcHist(
    [fimage],
    channels=[0],
    mask=None,
    histSize=[256],
    ranges=[0, 256]
)

counts_hist_2, bins_hist_2 = np.histogram(fimage, bins=256, range=(0, 256))
plt.hist(counts_hist_2, bins=256, range=(0, 256))
plt.show()
