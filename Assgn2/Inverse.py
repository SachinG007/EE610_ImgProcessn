import numpy as np # Import numpy for mathematical operations and functions
import sys # Required for starting and exiting application
import copy # Need the deepcopy function to copy entire arrays
from PIL import Image # Required to read jpeg images
import cv2 as cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def deblur(blur_img, kernel):

	rows,cols= np.shape(blur_img)
	pad_blur_img = np.zeros((2*rows, 2*cols), np.float64)
	pad_blur_img[0:blur_img.shape[0], 0:blur_img.shape[1]] = blur_img[:,:]
	rows,cols = np.shape(pad_blur_img)

	for i in range(rows):
		for j in range(cols):
			if((i+j)%2 == 1):
				pad_blur_img[i,j] = pad_blur_img[i,j] * (-1)

	#resize the kernel to the size of the gt image
	kernel = cv2.resize(kernel,(cols,rows))

	#taking the respective DFTs
	fft_blur_img = np.fft.fft2(pad_blur_img)
	fft_kernel = np.fft.fft2(kernel)
	# fft_kernel = np.fft.fftshift(fft_kernel)
	# fft_kernel = fft_kernel + 0.000001

	output = np.divide(fft_blur_img,np.abs(fft_kernel))
	output = np.fft.ifft2(output)
	output = (np.real(output)).astype(float)

	for i in range(rows):
		for j in range(cols):
			if((i+j)%2 == 1):
				output[i,j] = output[i,j] * (-1)

	crop_output = output[0:blur_img.shape[0], 0:blur_img.shape[1]]

	return crop_output


blur_img = cv2.imread("/Users/sachin007/Documents/EE610/EE610_ImgProcessn/Assgn2/blurred1.png",1) #load the blurred image
cv2.imshow("blurred_image", blur_img)
Rch,Gch,Bch = cv2.split(blur_img);	#split the image into the respective channels

kernel= cv2.imread("/Users/sachin007/Documents/EE610/EE610_ImgProcessn/Assgn2/kernel1b.png",1) #load the blurring filter 1 
cv2.imshow("kernel", kernel)
R_ker,G_ker,B_ker = cv2.split(blur_img); #split the kernel into the respective channels

R_deblur = deblur(Rch,R_ker)
R_deblur = R_deblur/np.max(R_deblur)
G_deblur = deblur(Gch,G_ker)
G_deblur = G_deblur/np.max(G_deblur)
B_deblur = deblur(Bch,B_ker)
B_deblur = B_deblur/np.max(B_deblur)

deblur_img = cv2.merge([R_deblur,G_deblur,B_deblur])
cv2.imshow("output",deblur_img)
cv2.waitKey(0)