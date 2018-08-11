# Python Starter Code Author: Devdatta Kathale

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget,QFileDialog, QInputDialog
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout



# from matplotlib.figure import Figure # Import matplotlib figure object
# #from matplotlib.backends.backend import FigureCanvasQTAgg as FigureCanvas 
# import matplotlib # Import matplotlib to set backend option
# matplotlib.use('QT5Agg') # Ensure using PyQt5 backend

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
import matplotlib.image as mpimg
np.set_printoptions(threshold='nan')

def conv2D(image,kernel):

    ker_x = kernel.shape[0]
    ker_y = kernel.shape[1]
    img_x = image.shape[0]
    img_y= image.shape[1]

    print(img_x)
    print(image[3][3])
    image = cv2.copyMakeBorder(image, 2,2,2,2, cv2.BORDER_DEFAULT, value = 0)
    print(image.shape)



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


class Window(QWidget):
    

    def __init__(self):
        
        super(Window, self).__init__()
        #Window x_cord,y_cord; Window size
        self.setGeometry(50, 50, 5000, 3000)
        #window heading
        self.setWindowTitle("Basic IP Tool")

        #define grid for auto layout setting, this makes managing the window quite easy 
        grid = QGridLayout()
        self.setLayout(grid)

        #defining button for loading an image
        btn_load = QPushButton("load_img", self)
        btn_load.clicked.connect(self.load_image)
        btn_load.resize(5,5)
        grid.addWidget(btn_load,0,0)

        #defining button for gamma correction
        btn_gamma = QPushButton("Gamma Correcn", self)
        val_gamma = btn_gamma.clicked.connect(self.load_gamma)
        btn_gamma.resize(5,5)
        grid.addWidget(btn_gamma,0,1)

        #defining button for Log Transform
        btn_log = QPushButton("Log Transform", self)
        btn_log.clicked.connect(self.log_transform)
        btn_log.resize(5,5)
        grid.addWidget(btn_log,1,0)

        #defining button for histogram Equalization
        btn_hist = QPushButton("Histogram Eq", self)
        btn_hist.clicked.connect(self.hist_eq)
        btn_hist.resize(5,5)
        grid.addWidget(btn_hist,1,1)

        #defining button for convolution 2d
        btn_blur = QPushButton("blur img", self)
        btn_blur.clicked.connect(self.blur_img)
        btn_blur.resize(5,5)
        grid.addWidget(btn_blur,2,0)


        #define figure and canvas to plot the loaded image on this
        self.figure = plt.figure(figsize=(15,5))
        self.canvas = FigureCanvas(self.figure)
        grid.addWidget(self.canvas,3,0,3,2)        
        self.show()

    def blur_img(self):
        kernel = (1/9)*np.matrix([[1, 1,1], [1, 1,1],[1,1,1]])
        conv2D(original_image_gray,kernel)

    def load_image(self):

        global original_image_gray

        ax = self.figure.add_subplot(221)
        ax.set_title("Original Image")
        filename = QFileDialog.getOpenFileName(self,'select')
        print(filename[0])
        original_image_gray = cv2.imread(str(filename[0]),0)
        print(original_image_gray.shape)
        # print(original_image_gray)
        print("dfs")
        plt.imshow(original_image_gray, cmap = "gray")
        self.canvas.draw()


    def load_gamma(self):

        gamma,ok = QInputDialog.getDouble(self,"Gamma Correction","Value of Gamma")

        if ok:

            gamma_corr_img = gamma_correction(original_image_gray,gamma)
            ax = self.figure.add_subplot(222)
            ax.set_title("Gamma Corrected Image")
            plt.imshow(gamma_corr_img, cmap = "gray")
            self.canvas.draw()


    def log_transform(self):

        c_for_log,ok = QInputDialog.getDouble(self,"Log Transformation","Value of C")

        if ok:
   
            log_transformed_img = c_for_log*(np.log10(1 + original_image_gray))
            #need to convert the value to interger becuse log has decimal values
            log_transformed_img = log_transformed_img.astype(int)
            # print(log_transformed_img)
            ax = self.figure.add_subplot(223)
            ax.set_title("Log Transformed Image")
            plt.imshow(log_transformed_img, cmap = "gray")
            self.canvas.draw()


    def hist_eq(self):
        histogram_output_img = histogram_eq(original_image_gray)

        ax = self.figure.add_subplot(222)
        ax.clear()
        plt.hist(original_image_gray.ravel(),256,[0,256]);
        ax.set_title("Histogram of Original Image")
        # plt.show()

        # print(histogram_output_img)
        ax = self.figure.add_subplot(224)
        ax.set_title("Histogram Equalization Output")
        plt.imshow(histogram_output_img, cmap = "gray")
        
        ax = self.figure.add_subplot(223)
        ax.clear()
        plt.hist(histogram_output_img.ravel(),256,[0,256]);
        ax.set_title("Histogram of New Image")
        # plt.show()

        self.canvas.draw()



def run():
    app = QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

run()