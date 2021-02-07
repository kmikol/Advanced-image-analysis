# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 08:38:06 2021

@author: zcq
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from matplotlib import pyplot as plt

def total_variation(img):
    gradients = np.gradient(img)
    area = img.shape[0]*img.shape[1]
    return np.sum(np.abs(gradients[0])+np.abs(gradients[1]))/area

def show_results(img, img_new):
    plt.Figure()
    plt.subplot(1,2,1)
    plt.title('original image')
    plt.imshow(img, cmap="gray")
    plt.subplot(1,2,2)
    plt.title('gaussian smoothing')
    plt.imshow(img_new, cmap="gray")
    print("the total variation of original picture is: ", total_variation(img))
    print("after gaussian smoothing, the total variation is: ", total_variation(img_new))
    
path_1 = 'data/fibres_xcth.png'
image_1 = imread(path_1)
img_1 = np.asarray(image_1)
img_1_gau = gaussian_filter(img_1, sigma=5)

path_2 = 'data/noisy_number.png'
image_2 = imread(path_2)
img_2 = np.asarray(image_2)
img_2_gau = gaussian_filter(img_2, sigma=5)

show_results(img_1, img_1_gau)

show_results(img_2, img_2_gau)