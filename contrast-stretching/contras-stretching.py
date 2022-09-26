#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 12:34:20 2022

@author: anhvietpham

@Document
https://khoatranrb.github.io/2020/02/21/cv-2
https://theailearner.com/2019/01/30/contrast-stretching/
https://pythontic.com/image-processing/pillow/contrast%20stretching
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

plt.rcParams['figure.figsize'] = [12,8]


original_image = cv2.imread("/Users/anhvietpham/Downloads/before_contrast_stretching.png")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

hst_red_image = cv2.calcHist(
    [original_image],
    [0],
    mask=None,
    histSize=[256], #full scale
    ranges=[0,256]
    )

hst_green_image = cv2.calcHist(
    [original_image],
    [1],
    mask=None,
    histSize=[256], #full scale
    ranges=[0,256]
    )

hst_blue_image = cv2.calcHist(
    [original_image],
    [2],
    mask=None,
    histSize=[256], #full scale
    ranges=[0,256]
    )


plt.subplot(4, 1, 1)
plt.imshow(original_image)
plt.title('image')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 1, 2)
#plt.plot(hst_red_image, color='r')
plt.stem(range(256), hst_red_image, use_line_collection=True)

plt.xlim([0, 255])
plt.title('red histogram')

plt.subplot(4, 1, 3)
plt.stem(range(256), hst_green_image, use_line_collection=True)
#plt.plot(hst_green_image, color='g')
plt.xlim([0, 255])
plt.title('green histogram')

plt.subplot(4, 1, 4)
#plt.plot(hst_blue_image, color='b')
plt.stem(range(256), hst_blue_image, use_line_collection=True)
plt.xlim([0, 255])
plt.title('blue histogram')

plt.tight_layout()
plt.show()

#r,g,b = cv2.split(original_image)

red_channel = original_image[:,:,0]
print(f"Min red channel {np.min(red_channel)}")
print(f"Max red channel {np.max(red_channel)}")

green_channel = original_image[:,:,1]
print(f"Min red channel {np.min(green_channel)}")
print(f"Max red channel {np.max(green_channel)}")

blue_channel = original_image[:,:,2]
print(f"Min red channel {np.min(blue_channel)}")
print(f"Max red channel {np.max(blue_channel)}")

plt.subplot(4, 1, 1)
plt.imshow(original_image)
plt.title('image')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 1, 2)
plt.imshow(red_channel, cmap=plt.cm.Reds_r)
plt.title('Red Channel')

plt.subplot(4, 1, 3)
plt.imshow(green_channel, cmap=plt.cm.Greens_r)
plt.title('Green Channel')

plt.subplot(4, 1, 4)
plt.imshow(blue_channel, plt.cm.Blues_r)
plt.title('Blue Channele')

plt.tight_layout()
plt.show()

def min_max_stretching(array):
    r, c = array.shape
    min_max_new_array = np.zeros((r, c), dtype = 'uint8')
    min_value = np.min(array)
    max_value = np.max(array)
    """
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            temp = (array[i,j] - min_value) / (max_value - min_value)
            min_max_new_array[i,j] = 255*temp
    """
    for i in range(r*c):
        temp = (array[i // c, i % c] - min_value) / (max_value - min_value)
        min_max_new_array[i // c, i % c] = 255*temp
    return min_max_new_array

min_max_red_channel = min_max_stretching(red_channel)
print(f"Min red channel {np.min(min_max_red_channel)}")
print(f"Max red channel {np.max(min_max_red_channel)}")


min_max_green_channel = min_max_stretching(green_channel)
print(f"Min red channel {np.min(min_max_green_channel)}")
print(f"Max red channel {np.max(min_max_green_channel)}")


min_max_blue_channel = min_max_stretching(blue_channel)
print(f"Min red channel {np.min(min_max_blue_channel)}")
print(f"Max red channel {np.max(min_max_blue_channel)}")


min_max_new_image = cv2.merge([min_max_red_channel, min_max_green_channel, min_max_blue_channel])

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(original_image)
plt.subplot(1,2,2)
plt.title("Min Max New Image")
plt.imshow(min_max_new_image)
plt.show()

hst_red_min_max_image = cv2.calcHist(
    [min_max_new_image],
    [0],
    mask=None,
    histSize=[256], #full scale
    ranges=[0,256]
    )

hst_green_min_max_image = cv2.calcHist(
    [min_max_new_image],
    [1],
    mask=None,
    histSize=[256], #full scale
    ranges=[0,256]
    )

hst_blue_min_max_image = cv2.calcHist(
    [min_max_new_image],
    [2],
    mask=None,
    histSize=[256], #full scale
    ranges=[0,256]
    )

plt.subplot(4, 1, 1)
plt.imshow(min_max_new_image)
plt.title('image')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 1, 2)
#plt.plot(hst_red_min_max_image, color='r')
plt.stem(range(256), hst_red_min_max_image, use_line_collection=True)

plt.xlim([0, 255])
plt.title('red histogram')

plt.subplot(4, 1, 3)
plt.stem(range(256), hst_green_min_max_image, use_line_collection=True)
#plt.plot(hst_green_min_max_image, color='g')
plt.xlim([0, 255])
plt.title('green histogram')

plt.subplot(4, 1, 4)
#plt.plot(hst_blue_min_max_image, color='b')
plt.stem(range(256), hst_blue_min_max_image, use_line_collection=True)
plt.xlim([0, 255])
plt.title('blue histogram')

plt.tight_layout()
plt.show()



