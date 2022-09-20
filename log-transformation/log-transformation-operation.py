#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 20:24:34 2022

@author: anhvietpham
https://viblo.asia/p/tuan-2-phep-toan-voi-diem-dieu-chinh-do-tuong-phan-V3m5WjpblO7
conda install spyder=5.3.3
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

plt.rcParams['figure.figsize'] = [12,8]

negative_original_image = cv2.imread('/Users/anhvietpham/Downloads/xyz.png')
log_transformation_dark_image = cv2.imread('/Users/anhvietpham/Downloads/log_image.png')
log_transformation_white_image = cv2.imread('/Users/anhvietpham/Downloads/log_bright_image.png')


# Plot quantization image of K level
def draw_quantization_img(levels, height=32):
    img = [levels] * height
    img = np.array(img)
    plt.imshow(img, 'gray')
    
    
# 256 level <=> 2^8 (8 bits)
gray_256 = list(range(0, 256, 1))
draw_quantization_img(gray_256)

# 64 level <=> 2^6 (6 bits)
gray_64 = list(range(0, 256, 4))
draw_quantization_img(gray_64)

# 32 level <=> 2^4 (4 bits)
gray_32 = list(range(0, 256, 8))
print(len(gray_32))
draw_quantization_img(gray_32, height=4)

# Image negative
negative_x_image = 255 - negative_original_image

plt.subplot(1,2,1)
plt.imshow(negative_original_image, cmap='gray')
plt.title("Original image")
plt.subplot(1, 2, 2)
plt.imshow(negative_x_image, cmap='gray')
plt.title("After using Negative Image")
plt.show()


# Log Transformation
c = 255 / np.log(1 + np.max(log_transformation_dark_image))

log_dark_image = c * (np.log(log_transformation_dark_image + 1))
log_dark_image = np.array(log_dark_image, dtype = np.uint8)

log_white_image = c * (np.log(log_dark_image + 1))
log_white_image = np.array(log_white_image, dtype = np.uint8)

log_white_image1 = c * (np.log(log_white_image + 1))
log_white_image1 = np.array(log_white_image1, dtype = np.uint8)


log_white_image2 = c * (np.log(log_white_image1 + 1))
log_white_image2 = np.array(log_white_image2, dtype = np.uint8)

log_white_image3 = c * (np.log(log_white_image2 + 1))
log_white_image3 = np.array(log_white_image3, dtype = np.uint8)

log_white_image4 = c * (np.log(log_white_image3 + 1))
log_white_image4 = np.array(log_white_image4, dtype = np.uint8)

log_white_image5 = c * (np.log(log_white_image4 + 1))
log_white_image5 = np.array(log_white_image5, dtype = np.uint8)

log_white_image6 = c * (np.log(log_white_image5 + 1))
log_white_image6 = np.array(log_white_image6, dtype = np.uint8)

log_white_image7 = c * (np.log(log_white_image6 + 1))
log_white_image7 = np.array(log_white_image7, dtype = np.uint8)

log_white_image8 = c * (np.log(log_white_image7 + 1))
log_white_image8 = np.array(log_white_image8, dtype = np.uint8)

log_white_image9 = c * (np.log(log_white_image8 + 1))
log_white_image9 = np.array(log_white_image9, dtype = np.uint8)

log_white_image10 = c * (np.log(log_white_image9 + 1))
log_white_image10 = np.array(log_white_image10, dtype = np.uint8)

plt.subplot(1,2,1)
plt.imshow(log_transformation_dark_image)
plt.title("Original image")
plt.subplot(1, 2, 2)
plt.imshow(log_dark_image)
plt.title("After using Log transformation Image")
plt.show()


plt.subplot(1,2,1)
plt.imshow(log_dark_image)
plt.title("Original image")
plt.subplot(1, 2, 2)
plt.imshow(log_white_image)
plt.title("After using Log transformation Image")
plt.show()

plt.subplot(1,2,1)
plt.imshow(log_white_image)
plt.title("Original image")
plt.subplot(1, 2, 2)
plt.imshow(log_white_image1)
plt.title("After using Log transformation Image")
plt.show()

plt.subplot(1,2,1)
plt.imshow(log_white_image1)
plt.title("Original image")
plt.subplot(1, 2, 2)
plt.imshow(log_white_image2)
plt.title("After using Log transformation Image")
plt.show()

plt.subplot(1,2,1)
plt.imshow(log_white_image2)
plt.title("Original image")
plt.subplot(1, 2, 2)
plt.imshow(log_white_image3)
plt.title("After using Log transformation Image")
plt.show()

plt.subplot(1,2,1)
plt.imshow(log_white_image2)
plt.title("Original image")
plt.subplot(1, 2, 2)
plt.imshow(log_white_image3)
plt.title("After using Log transformation Image")
plt.show()

