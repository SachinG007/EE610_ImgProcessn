# Python Starter Code Author: Devdatta Kathale

from PyQt4 import QtGui

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
from PyQt4.QtGui import *

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# np.set_printoptions(threshold='nan')



def gamma_correction(image,val_gamma):
    return np.power(image,val_gamma)

class Window(QtGui.QWidget):
    

    def __init__(self):
        
        super(Window, self).__init__()
        #Window x_cord,y_cord; Window size
        self.setGeometry(50, 50, 5000, 3000)
        #window heading
        self.setWindowTitle("Basic IP Tool")

        #define grid for auto layout setting, this makes managing the window quite easy 
        grid = QtGui.QGridLayout()
        self.setLayout(grid)

        #defining button for loading an image
        btn_load = QtGui.QPushButton("load_img", self)
        btn_load.clicked.connect(self.load_image)
        btn_load.resize(10,10)
        grid.addWidget(btn_load,0,0)

        #defining button for gamma correction
        btn_gamma = QtGui.QPushButton("Gamma Correcn", self)
        val_gamma = btn_gamma.clicked.connect(self.load_gamma)
        btn_gamma.resize(10,10)
        grid.addWidget(btn_gamma,0,1)

        #defining button for Log Transform
        btn_log = QtGui.QPushButton("Log Transform", self)
        btn_log.clicked.connect(self.log_transform)
        btn_log.resize(10,10)
        grid.addWidget(btn_log,0,2)


        #define figure and canvas to plot the loaded image on this
        self.figure = plt.figure(figsize=(15,5))
        self.canvas = FigureCanvas(self.figure)
        grid.addWidget(self.canvas,1,0,3,2)        
        self.show()

    def load_image(self):

        global original_image_gray

        ax = self.figure.add_subplot(221)
        ax.set_title("Original Image")
        filename = QtGui.QFileDialog.getOpenFileName(self,'select')
        original_image_gray = cv2.imread(str(filename),0)
        print(original_image_gray.shape)
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
            print(log_transformed_img)
            ax = self.figure.add_subplot(222)
            ax.set_title("Log Transformed Image")
            plt.imshow(log_transformed_img, cmap = "gray")
            self.canvas.draw()

def run():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

run()