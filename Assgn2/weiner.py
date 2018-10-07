import numpy as np # Import numpy for mathematical operations and functions
import sys # Required for starting and exiting application
import copy # Need the deepcopy function to copy entire arrays
from PIL import Image # Required to read jpeg images
import cv2 as cv2
import math
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import butter, lfilter, freqz

# Filter requirements.
order = 10
fs = 10000.0       # sample rate, Hz
cutoff = 70  # desired cutoff frequency of the filter, Hz

# def get_weiner(kernel,orig_img):





def deblur(blur_img, kernel,orig_img,shift_fft_mat):

	blur_img = blur_img.astype(float)
	kernel = kernel.astype(float)
	rows,cols= np.shape(blur_img)
	pad_blur_img = np.zeros((2*rows, 2*cols), np.float64)
	pad_blur_img[0:blur_img.shape[0], 0:blur_img.shape[1]] = blur_img[:,:]
	rows,cols = np.shape(pad_blur_img)

	pad_blur_img = np.multiply(pad_blur_img,shift_fft_mat)

	#resize the kernel to the size of the gt image
	kernel = cv2.resize(kernel,(cols,rows))
	

	#taking the respective DFTs
	fft_blur_img = np.fft.fft2(pad_blur_img)
	fft_kernel = np.fft.fft2(kernel)
	# fft_kernel = np.fft.fftshift(fft_kernel)
	# fft_kernel = fft_kernel + 0.000001

	output = np.divide(fft_blur_img,np.abs(fft_kernel))
	output = butter_lowpass_filter(output, cutoff, fs, order)
	# r,c = np.shape(output)
	# for i in range(500,r):
	# 	for j in range(500,c):
	# 		output[i,j]= 0;


	output = np.fft.ifft2(output)
	output = (np.real(output)).astype(float)
	output = np.multiply(output,shift_fft_mat)

	crop_output = output[0:blur_img.shape[0], 0:blur_img.shape[1]]

	return crop_output


orig_img = cv2.imread("/Users/sachin007/Documents/EE610/EE610_ImgProcessn/Assgn2/original1.jpg",1) #load the blurred image
orig_img = orig_img.astype(float)
Rch_orig,Gch_orig,Bch_orig = cv2.split(orig_img);	#split the image into the respective channels

blur_img = cv2.imread("/Users/sachin007/Documents/EE610/EE610_ImgProcessn/Assgn2/blurred1.jpg",1) #load the blurred image
cv2.imshow("blurred_image", blur_img)
blur_img = blur_img.astype(float)
Rch,Gch,Bch = cv2.split(blur_img);	#split the image into the respective channels

row_b,cols_b,chan = np.shape(blur_img)
shift_fft_mat = np.ones((row_b,cols_b))
for i in range(row_b):
	for j in range(cols_b):
		if((i+j)%2 == 1):
			shift_fft_mat[i,j] = shift_fft_mat[i,j] * (-1)

kernel= cv2.imread("/Users/sachin007/Documents/EE610/EE610_ImgProcessn/Assgn2/kernel1b.png",1) #load the blurring filter 1 
cv2.imshow("kernel", kernel)
R_ker,G_ker,B_ker = cv2.split(kernel); #split the kernel into the respective channels
kernel = kernel.astype(float)

R_deblur = deblur(Rch,R_ker,Rch_orig,shift_fft_mat)
R_deblur = R_deblur/np.max(R_deblur)
G_deblur = deblur(Gch,G_ker,Gch_orig,shift_fft_mat)
G_deblur = G_deblur/np.max(G_deblur)
B_deblur = deblur(Bch,B_ker,Bch_orig,shift_fft_mat)
B_deblur = B_deblur/np.max(B_deblur)

deblur_img = cv2.merge([R_deblur,G_deblur,B_deblur])
deblur_img = deblur_img.astype(float)
# import pdb;pdb.set_trace()
out_psnr = psnr(blur_img,deblur_img)
print(out_psnr)
cv2.imshow("output",deblur_img)
cv2.waitKey(0)