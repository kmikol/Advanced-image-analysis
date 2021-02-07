# Copy from jupyter notebook.
# I am not sure if it will display plots correctly in this format

import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import signal
import math as m
from scipy.sparse import diags

imgNoisy = cv2.imread('noisy_number.png',cv2.IMREAD_GRAYSCALE )

plt.figure(figsize = (8,8)) 
plt.imshow(img,cmap='gray')


# Convolve with a large gaussian to remove the noise
def createGaussianKernel(size,var):
    kern = np.zeros((size,size))
    tot = 0
    for x in range(-size//2+1,size//2+1):
        for y in range(-size//2+1,size//2+1):
            i = x+size//2
            j = y+size//2
            kern[i,j] =  m.exp(-(x**2+y**2)/(2*var))/(2*var*m.pi)
    return kern

imgDenoised = signal.convolve2d(imgNoisy,createGaussianKernel(25,25),mode='valid')
#imgDenoised = cv2.GaussianBlur(imgNoisy,(25,25),25)
plt.figure(figsize = (8,8)) 
plt.imshow(imgDenoised,cmap='gray')

# Segmentation boundary length
imgCell = cv2.imread('fuel_cells/fuel_cell_1.tif',cv2.IMREAD_GRAYSCALE )
plt.figure(figsize = (8,8)) 
plt.imshow(imgCell,cmap='gray')


kern = np.array([[-0.5,0.5]])
imgDiffX = np.abs(signal.convolve2d(imgCell,kern,mode='same'))
imgDiffY = np.abs(signal.convolve2d(imgCell,np.transpose(kern),mode='same'))
imgDiff = np.add(imgDiffX,imgDiffY)
imgDiff[imgDiff>0] = 1

# Subtract the first row and and column. The points are invalid
imgDiff = imgDiff[1:-1,1:-1]

count = np.count_nonzero(imgDiff)
print("The boundary length is {}".format(count))

plt.figure(figsize = (8,8)) 
plt.imshow(imgDiff,cmap='gray')


dinoNoisy = np.loadtxt('curves/dino_noisy.txt')
plt.plot(dinoNoisy[:,0], dinoNoisy[:,1], '-')

n = len(dinoNoisy)
k = np.array([np.ones(n-1),-2*np.ones(n),np.ones(n-1)])
offset = [-1,0,1]
L = diags(k,offset).toarray()
L[0,-1] = 1
L[-1,0] = 1

lamb = 0.25

#D = A*dinoNoisy
dinoDenoised = np.dot(np.add(np.eye(n),lamb*L),dinoNoisy)
#print(dinoDenoised)

plt.plot(dinoDenoised[:,0], dinoDenoised[:,1], '-')

# Compute curve length
distTot = 0
for i in range(-1,len(dinoDenoised)-1):
    
    p1 = dinoDenoised[i]
    p2 = dinoDenoised[i+1]
    dist = ((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**0.5
    distTot += dist
    
print("Total curve length: {}".format(distTot))
