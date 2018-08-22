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
        
        #defining a window for the gui        
        super(Window, self).__init__() 
        #Window x_cord,y_cord; Window size
        self.setGeometry(50, 50, 5000, 3000)
        #window heading
        self.setWindowTitle("Basic IP Tool")

        #define grid for auto layout setting, this makes managing the window quite easy 
        grid = QGridLayout()
        self.setLayout(grid)

        #defining button for loading an image
        #name of the button
        btn_load = QPushButton("load_img", self)
        #function to call when the button is pressed
        btn_load.clicked.connect(self.load_image)
        #size of the button
        btn_load.resize(5,5)
        #add the button to the layout
        grid.addWidget(btn_load,0,0)

        #defining button for gamma correction
        #name of the button
        btn_gamma = QPushButton("Gamma Correcn", self)
        val_gamma = btn_gamma.clicked.connect(self.load_gamma)
        btn_gamma.resize(5,5)
        grid.addWidget(btn_gamma,0,1)

        #defining button for Log Transform
        #name of the button
        btn_log = QPushButton("Log Transform", self)
        #function to call when the button is pressed
        btn_log.clicked.connect(self.log_transform)
        btn_log.resize(5,5)
        grid.addWidget(btn_log,1,0)

        #defining button for histogram Equalization
        #name of the button
        btn_hist = QPushButton("Histogram Eq", self)
        #function to call when the button is pressed
        btn_hist.clicked.connect(self.hist_eq)
        btn_hist.resize(5,5)
        grid.addWidget(btn_hist,1,1)

        #defining button for BLURR IMG
        #name of the button
        btn_blur = QPushButton("blur img", self)
        #function to call when the button is pressed
        btn_blur.clicked.connect(self.blur_img)
        btn_blur.resize(5,5)
        grid.addWidget(btn_blur,2,0)

        #defining button for GAUSSIAN bLUR
        #name of the button
        btn_sharp = QPushButton("gaussian blur", self)
        #function to call when the button is pressed
        btn_sharp.clicked.connect(self.gaussian_blur)
        btn_sharp.resize(5,5)
        grid.addWidget(btn_sharp,2,1)

        #defining button for sharpening
        #name of the button
        btn_sharp = QPushButton("sharp img", self)
        #function to call when the button is pressed
        btn_sharp.clicked.connect(self.sharp_img)
        btn_sharp.resize(5,5)
        grid.addWidget(btn_sharp,3,0)

        #defining button for go back to previous state
        #name of the button
        btn_sharp = QPushButton("Undo", self)
        #function to call when the button is pressed
        btn_sharp.clicked.connect(self.undo_prev)
        btn_sharp.resize(5,5)
        grid.addWidget(btn_sharp,7,0)

        #defining button for go back to original state
        #name of the button
        btn_sharp = QPushButton("Restore Original", self)
        #function to call when the button is pressed
        btn_sharp.clicked.connect(self.restore)
        btn_sharp.resize(5,5)
        grid.addWidget(btn_sharp,7,1)

        #defining button for storing thre image
        #name of the button
        btn_sharp = QPushButton("Save Image", self)
        #function to call when the button is pressed
        btn_sharp.clicked.connect(self.save_image)
        btn_sharp.resize(5,5)
        grid.addWidget(btn_sharp,8,0)

        #define figure and canvas to plot the loaded image on this
        self.figure = plt.figure(figsize=(15,5))
        self.canvas = FigureCanvas(self.figure)
        grid.addWidget(self.canvas,4,0,3,2)        
        # self.show()

        # self.h1figure = plt.figure(figsize=(15,5))
        # self.h1canvas = FigureCanvas(self.h1figure)
        # grid.addWidget(self.canvas,4,0,3,2)        
        self.show()


    def load_image(self):

        #popup the file explorere box, extract th file name
        filename = QFileDialog.getOpenFileName(self,'select')
        # print(filename[0])

        #read the selected file using mpimg
        self.original_image = mpimg.imread(str(filename[0]),1)
        # self.original_image = cv2.resize(self.original_image,(256,256))
        #convert the image to HSV, we will make changes only in the v channel of teh image
        self.hsv_image = cv2.cvtColor(self.original_image,cv2.COLOR_BGR2HSV)
        #extract the v_channel of the image for applying the transformations
        self.v_channel = self.hsv_image[:,:,2]
        #store the original image in another varibale, we may need in future to revert abck to original image
        self.v_channel_orig = np.copy(self.v_channel)


        # print(self.original_image.shape)
        #show the loaded image in a subplot
        ax = self.figure.add_subplot(121)
        #give a title to the image
        ax.set_title("Original Image")
        plt.imshow(self.original_image)
        self.canvas.draw()


    def load_gamma(self):
        #take the input value from the user for gamma
        gamma,ok = QInputDialog.getDouble(self,"Gamma Correction","Value of Gamma")

        if ok:
            #copy the present image into previous image variable for undo option
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
            #plot the image
            ax = self.figure.add_subplot(122)
            ax.set_title("Gamma Corrected Image")
            plt.imshow(output)
            self.canvas.draw()


    def log_transform(self):

        #user input for the C constant of the log functrion Clog(1+r)
        c_for_log,ok = QInputDialog.getDouble(self,"Log Transformation","Value of C")

        if ok:

            #copy the present image to previous for undo option
            self.v_channel_prev = np.copy(self.v_channel)            
            #maths equation  for log tranform
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
            #display the image
            ax = self.figure.add_subplot(122)
            ax.set_title("Log Transformed Image")
            plt.imshow(output)
            self.canvas.draw()


    def hist_eq(self):

        #copy the present image to previous for undo option
        self.v_channel_prev = np.copy(self.v_channel)            
        #first displaying thr histogram of the previous image
        # hax = self.h1figure.add_subplot(121)
        # ax.clear()
        # plt.hist(self.v_channel.ravel(),256,[0,256]);
        # hax.set_title("Histogram of Previous Image")
        # self.h1canvas.draw()
        
        #lets do the histogram equalization
        histogram_output_v = histogram_eq(self.v_channel)
        
        #normalize the outputs to the correect range in case they are out
        #not required in histogram equalization though
        hist_max = histogram_output_v.max()
        histogram_output_v = histogram_output_v * 255/hist_max
        
        #pass the changes to the hsv_image
        self.hsv_image[:,:,2] = histogram_output_v[:,:]
        self.v_channel = histogram_output_v[:,:]

        #convert the hsv image back to the RGB form for display
        output = cv2.cvtColor(self.hsv_image,cv2.COLOR_HSV2BGR)
        ax = self.figure.add_subplot(122)
        ax.set_title("Histogram Equalization Output")
        plt.imshow(output)
        
        # #histogram of the new image for comparison
        # ax = self.figure.add_subplot(224)
        # ax.clear()
        # plt.hist(self.v_channel.ravel(),256,[0,256]);
        # ax.set_title("Histogram of New Image")
        # # plt.show()

        # #Shw the original image also
        # self.hsv_image[:,:,2] = self.v_channel_orig[:,:]
        # output = cv2.cvtColor(self.hsv_image,cv2.COLOR_HSV2BGR)
        # ax = self.figure.add_subplot(221)
        # ax.set_title("Original Image")
        # plt.imshow(output)
        self.canvas.draw()


    def blur_img(self):

        #ask for the desired input size of the blurring kernel
        blur_size,ok = QInputDialog.getDouble(self,"Blur image","enter size of kernel (ODD ONLY")
        #if input is done
        if ok:   

            #copy the present image to previous for undo option
            self.v_channel_prev = np.copy(self.v_channel)            
            #convert input to integer
            blur_size= int(blur_size)

            #construct the basic blurring filter of the given input size
            kernel = (1/np.power(blur_size,2))*np.ones((blur_size,blur_size))
            #convolve the image with the blurring filter    
            blurred_v = conv2D(self.v_channel,kernel)
            #scale the values back to 0-255
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
            ax = self.figure.add_subplot(122)
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
                #copy the present image to previous for undo option
                self.v_channel_prev = np.copy(self.v_channel)            
                #convert the entered values to intergres
                kernel_size= int(kernel_size)
                sigma = int(sigma)

                #generate the appropriate gaussian filter using the inputs provided
                kernel = gen_gaussian(kernel_size,sigma)
                # print(kernel[2][2])
                #do the convolution
                blurred_v = conv2D(self.v_channel,kernel)
                #scale to 0-255
                blur_max = blurred_v.max()
                blurred_v = blurred_v * 255/blur_max

                #resizing the output, in convolution I have not done padding of the image    
                x = self.v_channel.shape[0]
                y = self.v_channel.shape[1]
                blurred_v = cv2.resize(blurred_v,(y,x))
                # print(blurred_v)

                #pass the changes to the hsv_image
                self.hsv_image[:,:,2] = blurred_v[:,:]
                self.v_channel = blurred_v[:,:]
                output = cv2.cvtColor(self.hsv_image,cv2.COLOR_HSV2BGR)
                ax = self.figure.add_subplot(122)
                ax.set_title("Gaussian blurred Image")
                plt.imshow(output)
                self.canvas.draw()




    def sharp_img(self):

        #ask for the value of A
        A_hboost,ok = QInputDialog.getInt(self,"Sharp image","enter A>1 for high boost sharpening")

        if ok:         
            
            self.v_channel_prev = np.copy(self.v_channel)            
            #construct filter to compute the laplacian 
            kernel = np.matrix([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
            laplace_sharped_v = conv2D(self.v_channel,kernel)
            # laplace_sharped_v = neg_pixel(laplace_sharped_v)
            min_intensity = laplace_sharped_v.min()
            laplace_sharped_v = laplace_sharped_v - min_intensity

            # #sobel filter to calculate the first gradient in x direction
            # kernel = np.matrix([[-1,-2,-1],[0,0,0],[1,2,1]])
            # Gx = conv2D(self.v_channel,kernel)
            # Gx = np.absolute(Gx)

            # #sobel filter to calculate the first gradient in y direction
            # kernel = np.matrix([[-1,0,1],[-2,0,2],[-1,0,1]])
            # Gy = conv2D(self.v_channel,kernel)
            # Gy = np.absolute(Gy)

            # #add the gradients
            # gradient = Gx + Gy

            # #the sum of the sobel gradients needs to be blurred as discussed in the lecture
            # kernel = (1/9)*np.matrix([[1,1,1],[1,1,1],[1,1,1]])
            # smooth_gradient = conv2D(gradient,kernel)
            # smooth_gradient = gradient

            # #resize the smooth gradient we computed to the original size so that we can multipky it with laplcian gradient image0
            # x = laplace_sharped_img.shape[0]
            # y = laplace_sharped_img.shape[1]
            # smooth_gradient = cv2.resize(smooth_gradient,(y,x))
            # max_smooth = smooth_gradient.max()
            # smooth_gradient = smooth_gradient * 255/max_smooth

            # #compute the final mask by multipling the 2 images
            # sharp_mask = np.multiply(smooth_gradient,laplace_sharped_img)
            # max_mask = sharp_mask.max()
            # sharp_mask = sharp_mask * 255/max_mask

            # #resize the msk to the input image shape
            # x = self.v_channel.shape[0]
            # y = self.v_channel.shape[1]
            # sharp_mask = cv2.resize(sharp_mask,(y,x))   

            # #add the mask and the input image
            # output_sharp = self.v_channel + sharp_mask
            # #apply power law transforms for more bttr results    
            # output_sharp_transformed = gamma_correction(output_sharp,0.6)
            # out_max = output_sharp_transformed.max()
            # output_sharp_transformed = output_sharp_transformed * 255/out_max

            #resize the msk to the input image shape
            x = self.v_channel.shape[0]
            y = self.v_channel.shape[1]
            laplace_sharped_v = cv2.resize(laplace_sharped_v,(y,x))   

            laplace_sharped_v = laplace_sharped_v + A_hboost*self.v_channel
            lap_max = laplace_sharped_v.max()
            laplace_sharped_v = laplace_sharped_v * 255/lap_max

            # laplace_sharped_v = laplace_sharped_v + self.v_channel

            self.hsv_image[:,:,2] = laplace_sharped_v[:,:]
            self.v_channel = laplace_sharped_v[:,:]

            output = cv2.cvtColor(self.hsv_image,cv2.COLOR_HSV2BGR)
            ax = self.figure.add_subplot(122)
            ax.set_title("sharpened Image")
            plt.imshow(output)

            self.canvas.draw()

    def undo_prev(self):
        self.v_channel_prev_temp = np.copy(self.v_channel_prev)
        self.v_channel_prev = np.copy(self.v_channel)
        self.v_channel = np.copy(self.v_channel_prev_temp)            

        self.hsv_image[:,:,2] = np.copy(self.v_channel[:,:])
        output = cv2.cvtColor(self.hsv_image,cv2.COLOR_HSV2BGR)
        ax = self.figure.add_subplot(122)
        ax.set_title("Previous Image")
        plt.imshow(output)
        self.canvas.draw()

    def restore(self):

        self.v_channel_prev = np.copy(self.v_channel)
        self.v_channel = np.copy(self.v_channel_orig)            
        self.hsv_image[:,:,2] = np.copy(self.v_channel[:,:])
        output = cv2.cvtColor(self.hsv_image,cv2.COLOR_HSV2BGR)
        ax = self.figure.add_subplot(122)
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