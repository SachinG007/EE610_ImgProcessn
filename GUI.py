# Python Starter Code Author: Devdatta Kathale

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget,QFileDialog, QInputDialog
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout
from GUI_functions import *


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


    def blur_img(self):

        blur_size,ok = QInputDialog.getDouble(self,"Blur image","enter size of kernel (ODD ONLY")

        if ok:   

            blur_size= int(blur_size)
            kernel = (1/np.power(blur_size,2))*np.ones((blur_size,blur_size))
            blurred_img = conv2D(original_image_gray,kernel)
            # print("outputshape: " )
            # print(blurred_img.shape)

            ax = self.figure.add_subplot(224)
            ax.set_title("blurred Image")
            plt.imshow(blurred_img, cmap = "gray")
            self.canvas.draw()


def run():
    app = QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

run()