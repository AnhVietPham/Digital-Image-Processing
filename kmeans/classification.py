#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 14:20:27 2022
@author: anhvietpham
https://www.unioviedo.es/compnum/labs/new/kmeans.html#:~:text=K%2Dmeans%20is%20an%20unsupervised,the%20group%20or%20cluster%20centroid.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from scipy.spatial.distance import cdist

n = 10

digits = load_digits()
data = digits.data
data = 255 - data

"""
kmeans = KMeans(n_clusters=n,init='random')
kmeans.fit(data)
Z = kmeans.predict(data)

for i in range(0,n):

    row = np.where(Z==i)[0]       # row in Z for elements of cluster i
    num = row.shape[0]            #  number of elements for each cluster
    r = int(np.floor(num/10.))    # number of rows in the figure of the cluster 

    print("cluster "+str(i))
    print(str(num)+" elements")

    plt.figure(figsize=(8,8))
    for k in range(0, num):
        plt.subplot(r+1, 10, k+1)
        image = data[row[k], ]
        image = image.reshape(8, 8)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.show()
"""

"""
plt.figure(figsize=(24,24))

for i in range(0, 20):
    plt.subplot(1, 20, i + 1)
    image = data[i]
    image = image.reshape(8,8)
    plt.imshow(image, cmap='gray')
plt.show()
"""


def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)]


def kmeans_assign_labels(X, centers):
    # calculate pairwise distances btw data and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis=1)


def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster
        Xk = X[labels == k, :]
        # take average
        centers[k, :] = np.mean(Xk, axis=0)
    return centers


def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) ==
            set([tuple(a) for a in new_centers]))


def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it)


(centers, labels, it) = kmeans(data, 10)
print('Centers found by our algorithm:')
print(centers[-1])
predict = labels[-1]

X = data[predict == 0, :]

row = int(np.floor(X.shape[0] / 10))

for i in range(0, 10):
    X = data[predict == i, :]
    row = int(np.floor(X.shape[0] / 10))
    for c in range(0, X.shape[0]):
        plt.subplot(row + 1, 10, c + 1)
        image = X[c]
        image = image.reshape(8, 8)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.show()
