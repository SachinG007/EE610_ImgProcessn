import numpy as np # Import numpy for mathematical operations and functions
import sys # Required for starting and exiting application
import copy # Need the deepcopy function to copy entire arrays
from PIL import Image # Required to read jpeg images
import cv2 as cv2
import math
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import butter, lfilter, freqz
import pdb;
import argparse
import pdb

def fft(img):

    rows , cols = np.shape(img)
    FFT = np.zeros(np.shape(img), dtype=complex)

    for i in range(rows):
        for j in range(cols):
            
            print(i,j)

            dft = complex(0)
            for m in range(rows):
                for n in range(cols):
                    dft = dft + img[n,m]*np.exp(-2j * np.pi * (m * i / rows + n*j/cols))
                    
            FFT[i,j] = dft
        
    return FFT

def new_fft(img):

    rows, cols = np.shape(img)
    FFT = np.zeros(np.shape(img), dtype = complex)

    for i in range(rows):
        for j in range(cols):

            print(i,j)

            x = np.arange(0,rows,1)/rows*i
            y = np.arange(0,cols,1)/cols*j
            xx, yy = np.meshgrid(x, y, sparse=True)
            z = xx + yy
            z = np.exp(z)
            z = np.power(z,-2j*np.pi)
            # pdb.set_trace()
            FFT[i,j] = np.trace(np.matmul(img,z))

    # FFT[FFT < np.power(.1,10)] = 0
    return FFT



blur_img = cv2.imread("/Users/sachin007/Documents/EE610/EE610_ImgProcessn/Assgn2/fft_sample.png",0) #load the blurred image
# r = 10
# c = 10
# # blur_img = cv2.resize(blur_img, dsize=(r,c))
cv2.imshow("ori",blur_img)

# pad_blur_img = np.zeros((2*r,2*c))
# pad_blur_img[0:r,0:c] = blur_img

# for i in range(2*r):
#   for j in range(2*c):
#       if((i+j)%2 == 1):
#           pad_blur_img[i,j] = pad_blur_img[i,j] * (-1)
pad_blur_img = blur_img

self_fft = new_fft(pad_blur_img)
mag_self_fft =  20 * np.log(np.abs(self_fft));
mag_self_fft[mag_self_fft < np.power(.1,10)] = 0

fft_np = (np.fft.fft2(blur_img))
mag_np_fft = 20 * np.log( np.abs(fft_np))
mag_np_fft[mag_np_fft < np.power(.1,10)] = 0
pdb.set_trace()


plt.subplot(221),plt.imshow(blur_img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(mag_np_fft, cmap = 'gray')
plt.title('Numpy Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(mag_self_fft, cmap = 'gray')
plt.title('Self Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
# pdb.set_trace()