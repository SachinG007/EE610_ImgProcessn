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


import sys
import os
from PyQt4.QtGui import *

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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
        btn_gamma.clicked.connect(self.load_gamma)
        btn_gamma.resize(10,10)
        grid.addWidget(btn_gamma,0,1)

        #define figure and canvas to plot the loaded image on this
        self.figure = plt.figure(figsize=(15,5))
        self.canvas = FigureCanvas(self.figure)
        grid.addWidget(self.canvas,1,0,3,2)        
        self.show()

    def load_gamma(self):
    	num,ok = QInputDialog.getInt(self,"Gamma Correction","Value of Gamma")
    	if ok:

    def load_image(self):
  	    ax = self.figure.add_subplot(222)
	    filename = QtGui.QFileDialog.getOpenFileName(self,'select')
	    image = mpimg.imread(str(filename))
	    plt.imshow(image)
	    self.canvas.draw()



def run():
	
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

run()
