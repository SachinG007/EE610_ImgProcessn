import numpy as np # Import numpy for mathematical operations and functions
import sys # Required for starting and exiting application
import copy # Need the deepcopy function to copy entire arrays
from PIL import Image # Required to read jpeg images
import cv2

import sys
import os
from PyQt5.QtGui import *

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def conv2D(image,kernel):

    kernel = np.flip(kernel,0)
    kernel = np.flip(kernel,1)
    ker_x = kernel.shape[0]
    ker_y = kernel.shape[1]
    kerx_by2 = (np.floor(ker_x/2)).astype(int)
    kery_by2 = (np.floor(ker_y/2)).astype(int)

    # print("input image shape")
    # print(image.shape)

    kernel_vector = np.transpose(np.reshape(kernel,(kernel.size,1)))

    img_width = image.shape[0]
    img_height = image.shape[1]

    #output will be smallr in size
    conv_image = np.zeros((img_width - ker_x + 1,img_height - ker_y + 1))

    # print(image[3][3])
    # pad_image = cv2.copyMakeBorder(image, kery_by2,kery_by2,kerx_by2,kerx_by2, cv2.BORDER_DEFAULT, value = 0)
    # pad_image = cv2.copyMakeBorder(image, 0,0,0,0, cv2.BORDER_DEFAULT, value = 0)
    pad_image = image

    print("pad_img shape : " )
    print(pad_image.shape)

    for i in range(0 + kerx_by2 , img_width - kerx_by2 ):
        for j in range(0 + kery_by2, img_height - kery_by2 ):

            current_patch = pad_image[i-kerx_by2 : i+kerx_by2+1 , j-kery_by2 : j+kery_by2+1]
            current_patch_vector = np.reshape(current_patch,(kernel.size,1))
            # print(current_patch_vector.shape)

            #copy the calculatedvalue to the output image
            conv_image[i-kerx_by2][j-kery_by2] = np.dot(kernel_vector,current_patch_vector)

    return conv_image


def gamma_correction(image,val_gamma):
    return np.power(image,val_gamma)

def histogram_eq(image):

    #first compute the frequncies of each of the intensity levels
    intensity_freq = np.zeros((256,1))

    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            intensity_val = image[i][j]
            intensity_freq[intensity_val] = intensity_freq[intensity_val] + 1;

    cumulative_freq = np.zeros((256,1))
    sum =0
    for k in range(0,256):
        sum = sum + intensity_freq[k]
        cumulative_freq[k] = sum

    total_pixels = image.shape[0] * image.shape[1]

    cumulative_freq_norm = 255*cumulative_freq/total_pixels    

    #construct the final image
    hist_eq_output_image = np.zeros((image.shape[0],image.shape[1]))

    for p in range(0,image.shape[0]):
        for q in range(0,image.shape[1]):
            hist_eq_output_image[p][q] = cumulative_freq_norm[image[p][q]]


    return hist_eq_output_image


def neg_pixel(image):

    min_intensity = image.min()
    image = image - min_intensity
    max_intensity = image.max()

    image = image * 255/max_intensity


    return image