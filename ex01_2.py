# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:41:56 2021

@author: zcq
"""

import numpy as np
from skimage.io import imread
from skimage.segmentation import find_boundaries

def task_0(img):
    img_boundary = find_boundaries(img)
    print("The length of segmentation boundary is {}.".format(np.sum(img_boundary!=0)))
    
def task_1(img):
    gradients = np.gradient(img)
    print("The length of segmentation boundary is {}.".format(np.sum((gradients[0]+gradients[1])!=0)//2))
    
path1 = 'data/fuel_cells/fuel_cell_1.tif'
image1 = imread(path1)
img1 = np.asarray(image1)

path2 = 'data/fuel_cells/fuel_cell_2.tif'
image2 = imread(path2)
img2 = np.asarray(image2)

path3 = 'data/fuel_cells/fuel_cell_3.tif'
image3 = imread(path3)
img3 = np.asarray(image3)

task_1(img1)
task_1(img2)
task_1(img3)
