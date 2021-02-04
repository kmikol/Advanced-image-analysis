# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:55:45 2021

@author: zcq
"""

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter, gaussian_filter1d
from skimage.io import imread
from matplotlib import pyplot as plt

def dnorm_derivative(x, mu, sd):
    front = -x/((sd**3)*np.sqrt(2*np.pi)) 
    bag = np.e**(-(1/2)*(np.power((x-mu)/sd,2)))
    pdf = front*bag
    return pdf

def gaussian_kernel_derivate(size, show=False):
    sigma = np.sqrt(size)
    kernel_1D = np.linspace(-(size //        2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm_derivative(kernel_1D[i], 0, sigma)
    # print('kernel 1D {}'.format(kernel_1D))

    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max() # scales the kernel up

    # print('kernel 2D {}'.format(kernel_2D))

    if show:
        plt.imshow(kernel_2D, interpolation = 'none', cmap = 'gray')
        plt.title('image')
        plt.show()
    return kernel_2D

def task_1(img):
    img_gau_1d = gaussian_filter1d(img, sigma=5)
    img_gau_1d = gaussian_filter1d(img_gau_1d.T, sigma=5)
    img_gau_1d = img_gau_1d.T
    img_gau_2d = gaussian_filter(img, sigma=5)
    
    plt.Figure()
    plt.subplot(1,3,1)
    plt.title('1D gauss kernel')
    plt.imshow(img_gau_1d)
    plt.imshow(img, cmap="gray")
    plt.subplot(1,3,2)
    plt.title('2D gauss kernel')
    plt.imshow(img_gau_2d, cmap="gray")  
    plt.subplot(1,3,3)
    plt.title('difference between 1D and 2D')
    diff1 = img_gau_1d - img_gau_2d
    plt.imshow(diff1, cmap="gray")
    print('standard variance of the subtractet images : {}'.format(np.std(diff1)))

def task_2(img):
    #k = np.array([[0.5,0,-0.5],[0.5,0,-0.5],[0.5,0,-0.5]])
    k = np.array([0.5, 0, -0.5])
    res1 = convolve1d(gaussian_filter1d(img, sigma=5), k, axis=1)
    res2 = gaussian_filter1d(img, sigma=5, order=1)
    ## show the results
    plt.Figure()
    plt.subplot(1,3,1)
    plt.title('d(I*g)/dx')
    plt.imshow(res1, cmap="gray")
    plt.subplot(1,3,2)
    plt.title('I*dg/dx')
    plt.imshow(res2, cmap="gray")  
    plt.subplot(1,3,3)
    plt.title('d(I*g)/dx - I*dg/dx')
    diff2 = res1 - res2
    plt.imshow(diff2, cmap="gray") 
    print('standard variance of the subtractet images : {}'.format(np.std(diff2)))

def task_3(img):
    res1 = gaussian_filter(img, sigma=20)
    res2 = gaussian_filter(img, sigma=2)
    for i in np.arange(9):
        res2 = gaussian_filter(res2, sigma=2)
    plt.Figure()
    plt.subplot(1,3,1)
    plt.title('sigma=20')
    plt.imshow(res1, cmap="gray")
    plt.subplot(1,3,2)
    plt.title('sigma=2 * 10')
    plt.imshow(res2, cmap="gray")  
    plt.subplot(1,3,3)
    plt.title('difference')
    diff3 = res1 - res2
    plt.imshow(diff3, cmap="gray") 
    print('standard variance of the subtractet images : {}'.format(np.std(diff3)))

def task_4(img):
    res1 = gaussian_filter1d(img, sigma=np.sqrt(20), order=1)
    res2 = gaussian_filter(img, sigma=np.sqrt(10))
    res2 = gaussian_filter1d(res2, sigma=np.sqrt(10), order=1)
    plt.Figure()
    plt.subplot(1,3,1)
    plt.title('res1')
    plt.imshow(res1, cmap="gray")
    plt.subplot(1,3,2)
    plt.title('res2')
    plt.imshow(res2, cmap="gray")  
    plt.subplot(1,3,3)
    plt.title('difference')
    diff4 = res1 - res2
    plt.imshow(diff4, cmap="gray") 
    print('standard variance of the subtractet images : {}'.format(np.std(diff4)))
    
#path = 'data/noisy_number.png'
path = 'data/fibres_xcth.png'
image = imread(path)
img = np.asarray(image)
#task_1(img)
#task_2(img)
#task_3(img)
#task_4(img)