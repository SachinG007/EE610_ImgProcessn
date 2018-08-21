# Python Starter Code Author: Devdatta Kathale

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget,QFileDialog, QInputDialog
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout
from GUI_functions import *
# from copy import deepcopy

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

# original_image = np.zeros((256,256))
# hsv_image = np.zeros((256,256))
# v_channel = np.zeros((256,256))
# v_channel_prev = np.zeros((256,256))

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

        #defining button for BLURR IMG
        btn_blur = QPushButton("blur img", self)
        btn_blur.clicked.connect(self.blur_img)
        btn_blur.resize(5,5)
        grid.addWidget(btn_blur,2,0)

        #defining button for GAUSSIAN bLUR
        btn_sharp = QPushButton("gaussian blur", self)
        btn_sharp.clicked.connect(self.gaussian_blur)
        btn_sharp.resize(5,5)
        grid.addWidget(btn_sharp,2,1)

        #defining button for sharpening
        btn_sharp = QPushButton("sharp img", self)
        btn_sharp.clicked.connect(self.sharp_img)
        btn_sharp.resize(5,5)
        grid.addWidget(btn_sharp,3,0)

        #defining button for go back to previous state
        btn_sharp = QPushButton("Undo", self)
        btn_sharp.clicked.connect(self.undo_prev)
        btn_sharp.resize(5,5)
        grid.addWidget(btn_sharp,3,1)

        #defining button for go back to original state
        btn_sharp = QPushButton("Restore Original", self)
        btn_sharp.clicked.connect(self.restore)
        btn_sharp.resize(5,5)
        grid.addWidget(btn_sharp,4,0)

        #defining button for storing thre image
        btn_sharp = QPushButton("Save Image", self)
        btn_sharp.clicked.connect(self.save_image)
        btn_sharp.resize(5,5)
        grid.addWidget(btn_sharp,4,1)

        #define figure and canvas to plot the loaded image on this
        self.figure = plt.figure(figsize=(15,5))
        self.canvas = FigureCanvas(self.figure)
        grid.addWidget(self.canvas,5,0,3,2)        
        self.show()


    def load_image(self):


        filename = QFileDialog.getOpenFileName(self,'select')
        print(filename[0])

        self.original_image = mpimg.imread(str(filename[0]),1)
        # self.original_image = cv2.resize(self.original_image,(256,256))
        self.hsv_image = cv2.cvtColor(self.original_image,cv2.COLOR_BGR2HSV)
        self.v_channel = self.hsv_image[:,:,2]
        self.v_channel_orig = np.copy(self.v_channel)


        print(self.original_image.shape)
        ax = self.figure.add_subplot(221)
        ax.set_title("Original Image")
        plt.imshow(self.original_image)
        self.canvas.draw()


    def load_gamma(self):

        gamma,ok = QInputDialog.getDouble(self,"Gamma Correction","Value of Gamma")

        if ok:
            self.v_channel_prev = np.copy(self.v_channel)
            # pass the v_channel to the gamma correction function    
            gamma_corr_v = gamma_correction(self.v_channel,gamma)
            # since we are applying a power operation, we need to scale back all the values to 0-255
            gamma_max = gamma_corr_v.max()
            gamma_corr_v = gamma_corr_v * 255 /gamma_max
            #update the v channel
            self.v_channel = gamma_corr_v[:,:]
            #update teh v_channel in the hsv image
            self.hsv_image[:,:,2] = self.v_channel[:,:]

            #convert the hsv image back to the RGB form for display    
            output = cv2.cvtColor(self.hsv_image,cv2.COLOR_HSV2BGR)
            ax = self.figure.add_subplot(222)
            ax.set_title("Gamma Corrected Image")
            plt.imshow(output)
            self.canvas.draw()


    def log_transform(self):

        c_for_log,ok = QInputDialog.getDouble(self,"Log Transformation","Value of C")

        if ok:

            self.v_channel_prev = np.copy(self.v_channel)            
            #maths equation   
            log_transformed_v = c_for_log*(np.log10(1 + self.v_channel))
            #need to convert the value to interger becuse log has decimal values
            log_transformed_v = log_transformed_v.astype(int)
            #scale bacak to 0-255
            log_max = log_transformed_v.max()
            log_transformed_v = log_transformed_v * 255/log_max
            #take care of the negative pixel values
            log_transformed_v = neg_pixel(log_transformed_v)
            #add the cahnges to the hsv_image
            self.hsv_image[:,:,2] = log_transformed_v[:,:]
            self.v_channel = log_transformed_v[:,:]

            #convert the hsv image back to the RGB form for display
            output = cv2.cvtColor(self.hsv_image,cv2.COLOR_HSV2BGR)
            ax = self.figure.add_subplot(222)
            ax.set_title("Log Transformed Image")
            plt.imshow(output)
            self.canvas.draw()


    def hist_eq(self):

        self.v_channel_prev = np.copy(self.v_channel)            
        #first displaying thr histogram of the original image
        ax = self.figure.add_subplot(222)
        ax.clear()
        plt.hist(self.v_channel.ravel(),256,[0,256]);
        ax.set_title("Histogram of Original Image")

        #lets do the histogram equalization
        histogram_output_v = histogram_eq(self.v_channel)
        
        hist_max = histogram_output_v.max()
        histogram_output_v = histogram_output_v * 255/hist_max
        
        self.hsv_image[:,:,2] = histogram_output_v[:,:]
        self.v_channel = histogram_output_v[:,:]

        #convert the hsv image back to the RGB form for display
        output = cv2.cvtColor(self.hsv_image,cv2.COLOR_HSV2BGR)
        ax = self.figure.add_subplot(224)
        ax.set_title("Histogram Equalization Output")
        plt.imshow(output)
        
        #histogram of the new image for comparison
        ax = self.figure.add_subplot(223)
        ax.clear()
        plt.hist(self.v_channel.ravel(),256,[0,256]);
        ax.set_title("Histogram of New Image")
        # plt.show()
        self.canvas.draw()


    def blur_img(self):

        #ask for the desired input size of the blurring kernel
        blur_size,ok = QInputDialog.getDouble(self,"Blur image","enter size of kernel (ODD ONLY")
        #if input is done
        if ok:   

            self.v_channel_prev = np.copy(self.v_channel)            
            #convert input to integer
            blur_size= int(blur_size)

            #construct the basic blurring filter of the given input size
            kernel = (1/np.power(blur_size,2))*np.ones((blur_size,blur_size))
            #convol;ve the image with the blurring filter    
            blurred_v = conv2D(self.v_channel,kernel)
            blur_max = blurred_v.max()
            blurred_v = blurred_v * 255/blur_max

            #resizing the output, in convolution I have not done padding of the image    
            x = self.v_channel.shape[0]
            y = self.v_channel.shape[1]
            blurred_v = cv2.resize(blurred_v,(y,x))
            # print(blurred_v)

            self.hsv_image[:,:,2] = blurred_v[:,:]
            self.v_channel = blurred_v[:,:]
            #convert the hsv image back to the RGB form for display   
            output = cv2.cvtColor(self.hsv_image,cv2.COLOR_HSV2BGR)
            ax = self.figure.add_subplot(224)
            ax.set_title("blurred Image")
            plt.imshow(output)
            self.canvas.draw()

    def gaussian_blur(self):

        #implementation of the gaussian blurring
        #ask for the kernel size
        kernel_size,ok = QInputDialog.getDouble(self,"Gaussian Blur image","enter size of kernel (ODD ONLY")
        if ok:   

            #ask for the sigma value of the gaussian function
            sigma,ok = QInputDialog.getDouble(self,"Gaussian Blur image","enter vale of sigma")
            if ok:   

                self.v_channel_prev = np.copy(self.v_channel)            
                #convert the entered values to intergres
                kernel_size= int(kernel_size)
                sigma = int(sigma)

                #generate the appropriate gaussian filter using the inputs provided
                kernel = gen_gaussian(kernel_size,sigma)
                # print(kernel[2][2])
                #do the convolution
                blurred_v = conv2D(self.v_channel,kernel)
                blur_max = blurred_v.max()
                blurred_v = blurred_v * 255/blur_max

                x = self.v_channel.shape[0]
                y = self.v_channel.shape[1]
                blurred_v = cv2.resize(blurred_v,(y,x))
                # print(blurred_v)

                self.hsv_image[:,:,2] = blurred_v[:,:]
                self.v_channel = blurred_v[:,:]
                output = cv2.cvtColor(self.hsv_image,cv2.COLOR_HSV2BGR)
                ax = self.figure.add_subplot(224)
                ax.set_title("Gaussian blurred Image")
                plt.imshow(output)
                self.canvas.draw()




    def sharp_img(self):

        #ask for the value of A
        A_hboost,ok = QInputDialog.getInt(self,"Sharp image","enter A>1 for high boost sharpening")

        if ok:         

            self.v_channel_prev = np.copy(self.v_channel)            
            #construct filter to compute the laplacian 
            kernel = np.matrix([[-1,-1,-1],[-1,A_hboost + 8,-1],[-1,-1,-1]])
            laplace_sharped_img = conv2D(self.v_channel,kernel)
            laplace_sharped_img = neg_pixel(laplace_sharped_img)
            
            #sobel filter to calculate the first gradient in x direction
            kernel = np.matrix([[-1,-2,-1],[0,0,0],[1,2,1]])
            Gx = conv2D(self.v_channel,kernel)
            Gx = np.absolute(Gx)

            #sobel filter to calculate the first gradient in y direction
            kernel = np.matrix([[-1,0,1],[-2,0,2],[-1,0,1]])
            Gy = conv2D(self.v_channel,kernel)
            Gy = np.absolute(Gy)

            #add the gradients
            gradient = Gx + Gy

            #the sum of the sobel gradients needs to be blurred as discussed in the lecture
            kernel = (1/9)*np.matrix([[1,1,1],[1,1,1],[1,1,1]])
            smooth_gradient = conv2D(gradient,kernel)
            smooth_gradient = gradient

            #resize the smooth gradient we computed to the original size so that we can multipky it with laplcian gradient image0
            x = laplace_sharped_img.shape[0]
            y = laplace_sharped_img.shape[1]
            smooth_gradient = cv2.resize(smooth_gradient,(y,x))
            max_smooth = smooth_gradient.max()
            smooth_gradient = smooth_gradient * 255/max_smooth

            #compute the final mask by multipling the 2 images
            sharp_mask = np.multiply(smooth_gradient,laplace_sharped_img)
            max_mask = sharp_mask.max()
            sharp_mask = sharp_mask * 255/max_mask

            #resize the msk to the input image shape
            x = self.v_channel.shape[0]
            y = self.v_channel.shape[1]
            sharp_mask = cv2.resize(sharp_mask,(y,x))   

            #add the mask and the input image
            output_sharp = self.v_channel + sharp_mask
            #apply power law transforms for more bttr results    
            output_sharp_transformed = gamma_correction(output_sharp,0.6)
            out_max = output_sharp_transformed.max()
            output_sharp_transformed = output_sharp_transformed * 255/out_max

            self.hsv_image[:,:,2] = smooth_gradient[:,:]
            self.v_channel = output_sharp_transformed[:,:]

            output = cv2.cvtColor(self.hsv_image,cv2.COLOR_HSV2BGR)
            ax = self.figure.add_subplot(111)
            ax.set_title("sharpened Image")
            plt.imshow(output)

            self.canvas.draw()

    def undo_prev(self):
        self.v_channel_prev_temp = np.copy(self.v_channel_prev)
        self.v_channel_prev = np.copy(self.v_channel)
        self.v_channel = np.copy(self.v_channel_prev_temp)            

        self.hsv_image[:,:,2] = np.copy(self.v_channel[:,:])
        output = cv2.cvtColor(self.hsv_image,cv2.COLOR_HSV2BGR)
        ax = self.figure.add_subplot(222)
        ax.set_title("Previous Image")
        plt.imshow(output)
        self.canvas.draw()

    def restore(self):

        self.hsv_image[:,:,2] = np.copy(self.v_channel_orig[:,:])
        output = cv2.cvtColor(self.hsv_image,cv2.COLOR_HSV2BGR)
        ax = self.figure.add_subplot(222)
        ax.set_title("Original Image")
        plt.imshow(output)
        self.canvas.draw()

    def save_image(self):

        self.hsv_image[:,:,2] = self.v_channel[:,:]
        output = cv2.cvtColor(self.hsv_image,cv2.COLOR_HSV2RGB)
        cv2.imwrite("Current Image.jpg",output)

def run():
    app = QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

run()