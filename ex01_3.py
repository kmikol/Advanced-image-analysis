# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:19:25 2021

@author: zcq
"""

import numpy as np
from scipy.ndimage import convolve, convolve1d, gaussian_filter, gaussian_filter1d, gaussian_gradient_magnitude
import scipy.linalg
from scipy.interpolate import interp2d
from skimage.io import imread
from matplotlib import pyplot as plt

plt.figure()
dino = np.loadtxt("data/curves/dino.txt")
dino_noisy = np.loadtxt("data/curves/dino_noisy.txt")
hand = np.loadtxt("data/curves/hand.txt")
hand_noisy = np.loadtxt("data/curves/hand_noisy.txt")

#plt.plot(dino[:, 0], dino[:, 1], dino_noisy[:, 0], dino_noisy[:, 1])
plt.plot(hand[:, 0], hand[:, 1], hand_noisy[:, 0], hand_noisy[:, 1])

def matrix_L(n, val):
    L = np.zeros((n,n))
    for i in np.arange(n):
        for j in np.arange(n):
            if (i==j):
                L[i,j] = val
            elif (i == j+1 or i+1 == j):
                L[i,j] = 1
    L[n-1,0] = 1
    L[0,n-1] = 1
    return L


def task_1(points, true, lamb):
    N, M = points.shape
    I = np.identity(N)
    points_new = (I + lamb * matrix_L(N, -2)).dot(points)
    plt.figure()
    plt.plot(true[:, 0], true[:, 1], points_new[:, 0], points_new[:, 1])
    
def task_2(points, true, lamb):
    N, M = points.shape
    I = np.identity(N)
    points_new = np.linalg.inv(I - lamb * matrix_L(N, -2)).dot(points)
    plt.figure()
    plt.plot(true[:, 0], true[:, 1], points_new[:, 0], points_new[:, 1])
    
def task_3(points, true, lamb, a, b):
    N, M = points.shape
    I = np.identity(N)
    A = matrix_L(N, -2)
    B = matrix_L(N, -6)
    points_new = np.linalg.inv(I - a * A - b * B).dot(points)
    plt.figure()
    plt.plot(true[:, 0], true[:, 1], points_new[:, 0], points_new[:, 1])
    

    
#task_1(dino_noisy, dino, 0.5)
#task_2(dino_noisy, dino, 0.5)
#task_3(dino_noisy, dino, lamb = 0.5, a = 5, b = -0.01)

task_1(hand_noisy, hand, 0.5)
task_2(hand_noisy, hand, 0.5)
task_3(hand_noisy, hand, lamb = 0.5, a = 5, b = -0.01)