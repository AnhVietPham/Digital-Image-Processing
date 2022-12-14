"""
Author: Anh Viet Pham
https://medium.com/@akumar5/computer-vision-gaussian-filter-from-scratch-b485837b6e09
"""

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import argparse

"""
https://www.youtube.com/watch?v=f0t-OCG79-U
"""

def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:
        print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size : {}".format(image.shape))
    else:
        print("Image Shape : {}".format(image.shape))

    print("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    print("Output Image size : {}".format(output.shape))

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()

    return output


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(l=3, sig=10):
    # kernel_1D = np.linspace(-(size // 2), size // 2, size)
    # for i in range(size):
    #     kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    # kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    #
    # kernel_2D *= 1.0 / kernel_2D.max()
    #
    # if verbose:
    #     plt.imshow(kernel_2D, interpolation='none', cmap='gray')
    #     plt.title("Kernel ( {}X{} )".format(size, size))
    #     plt.show()

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    kernel = (1 / (2 * math.pi * np.square(sig))) * kernel

    return kernel / np.sum(kernel)


def gaussian_blur(image, kernel_size, verbose=False):
    kernel = gaussian_kernel(kernel_size, sig=10)
    return convolution(image, kernel, average=True, verbose=verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",
                        default="/Users/sendo_mac/Documents/avp/Digital-Image-Processing/spatial-filtering/data/bird.png",
                        help="Path to the image")
    args = parser.parse_args()

    image = cv2.imread(args.image)

    gaussian_blur(image, 5, verbose=True)
