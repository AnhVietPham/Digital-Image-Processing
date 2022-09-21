#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 07:50:35 2022

@author: anhvietpham
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

plt.rcParams['figure.figsize'] = [12,8]

image = cv2.imread("/Users/anhvietpham/Downloads/test_histogram.jpeg")

red_channel = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
red_width, red_height = image[:,:,0].shape
normalize_histogram_red_channel = red_channel / (red_width*red_height)

np_red_channel = np.histogram(image[:,:, 0], bins=256, range=(0, 256))[0]


def calcHist(img):
  hist_image = [cv2.calcHist([img],[i], None, [256], [0,256]).flatten() for i in range(3)]
  hist_array_image = np.array(hist_image).ravel()  
  return hist_array_image.astype(np.uint8)


innames = glob.glob("/Users/anhvietpham/Downloads/INT3404_20-master/week3/code/flowers/*.jpg")
innames = sorted(innames)
print(innames)

images = [cv2.imread(inname) for inname in innames]

xs = [calcHist(img) for img in images]

test_input_image_histogram = calcHist(image)

image_input = images[0]
xs_input = xs[0]

all_distances = [np.linalg.norm(test_input_image_histogram - xsi) for xsi in xs]

sorted_idxs = np.argsort(all_distances)
print(sorted_idxs)

ranked_images = [images[i] for i in sorted_idxs]


plt.subplot(2, 3, 1)
plt.imshow(image[:,:,::-1])
plt.title("Input")
for i in range(0, 5):
    plt.subplot(2, 3, i+2)
    idx = sorted_idxs[i]
    found = ranked_images[i]
    dist = all_distances[idx]
    plt.imshow(found[:,:,::-1])
    plt.title("Rank #{}, distance={:.2f}".format(i, dist))
plt.show()


