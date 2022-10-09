#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 13:55:40 2022

@author: anhvietpham
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

image = cv2.imread("/Users/anhvietpham/Downloads/pepper.jpeg")


def rgb_filter(image, kernel, mode='full', boundary='fill'):
    r, g, b = cv2.split(image[:, :, ::-1])
    r = signal.correlate2d(r, kernel, mode=mode, boundary=boundary)
    g = signal.correlate2d(g, kernel, mode=mode, boundary=boundary)
    b = signal.correlate2d(b, kernel, mode=mode, boundary=boundary)
    output = cv2.merge([r, g, b])
    return output


kernel = np.zeros((3, 3), dtype='uint8')
kernel[1, 1] = 1
filter_image = rgb_filter(image, kernel)

print('Original Image', image.shape)
print('Filterd Image', filter_image.shape)

plt.subplot(1, 2, 1)
plt.imshow(image[:, :, ::-1])
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filter_image)
plt.axis('off')
plt.show()

"""
Image Size:  M x N 
Kernel Size: m x n

==> Output : (M + m - 1) x (N + n - 1)
"""
random_image = np.random.random([5, 5])
random_image = random_image * 255

filter_random_image = signal.correlate2d(random_image, kernel, mode='full', boundary='fill')

plt.subplot(1, 2, 1)
plt.imshow(random_image, cmap='gray')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filter_random_image, cmap='gray')
plt.colorbar()
plt.axis('off')
plt.show()

"""
Image Size:  M x N 
Kernel Size: m x n

==> Output : (M - n + 1) x (M - n + 1)
"""

random_image_mode_valid = np.random.random([5, 5])
random_image_mode_valid = random_image_mode_valid * 255

filter_random_image_mode_valid = signal.correlate2d(random_image_mode_valid, kernel, mode='valid', boundary='fill')

plt.subplot(1, 2, 1)
plt.imshow(random_image_mode_valid, cmap='gray')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filter_random_image_mode_valid, cmap='gray')
plt.colorbar()
plt.axis('off')
plt.show()

"""
Image Size:  M x N 
Kernel Size: m x n

==> Output : M  x N
"""

random_image_mode_same = np.random.random([5, 5])
random_image_mode_same = random_image_mode_same * 255

filter_random_image_mode_same = signal.correlate2d(random_image_mode_same, kernel, mode='same', boundary='fill')

plt.subplot(1, 2, 1)
plt.imshow(random_image_mode_same, cmap='gray')
plt.colorbar()
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filter_random_image_mode_same, cmap='gray')
plt.colorbar()
plt.axis('off')
plt.show()
