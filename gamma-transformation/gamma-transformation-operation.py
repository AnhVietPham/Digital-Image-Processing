#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:43:13 2022

@author: anhvietpham
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

original_image = cv2.imread('/Users/anhvietpham/Downloads/gamma-low-exposure.jpeg')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
convert_BGR2RGB = original_image[:,:,::-1]
gamma_image = np.array(255*(original_image/255)**0.4, dtype='uint8')

plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original image")
plt.subplot(1, 2, 2)
plt.imshow(gamma_image)
plt.title("After using Gamma transformation")
plt.show()
