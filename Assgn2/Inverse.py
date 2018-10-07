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
cutoff = 430  # desired cutoff frequency of the filter, Hz
cutoff_radius = 200
r_sq = cutoff_radius**2

def psnr(img1, img2):	#input,output
    mse = np.mean( (img1 - img2) ** 2 ).astype(float)
    inp_max = np.max(img1)
    return 20 * math.log10(inp_max / math.sqrt(mse))
    # return mse

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def deblur(blur_img, kernel,shift_fft_mat):

	blur_img = blur_img.astype(float)
	kernel = kernel.astype(float)

	rows,cols= np.shape(blur_img)
	kernel = cv2.resize(kernel,(cols,rows))


	pad_blur_img = np.zeros((2*rows, 2*cols), np.float64)
	pad_kernel = np.zeros((2*rows, 2*cols), np.float64)
	pad_blur_img[0:blur_img.shape[0], 0:blur_img.shape[1]] = blur_img[:,:]
	pad_kernel[0:blur_img.shape[0], 0:blur_img.shape[1]] = kernel[:,:]

	pad_blur_img = np.multiply(pad_blur_img,shift_fft_mat)
	pad_kernel = np.multiply(pad_kernel,shift_fft_mat)



	#taking the respective DFTs
	fft_blur_img = np.fft.fft2(pad_blur_img)
	fft_kernel = np.fft.fft2(pad_kernel)

	output = np.divide(fft_blur_img,fft_kernel)

	r,c = np.shape(output)
	c0 = r/2
	c1= c/2
	for i in range(r):
		for j in range(c):
			if ((i-c0)**2 + (j-c1)**2 < r_sq):
				output[i,j]= fft_blur_img[i,j];


	output = np.fft.ifft2(output)
	output = (np.real(output)).astype(float)

	output = np.multiply(output,shift_fft_mat)

	crop_output = output[0:blur_img.shape[0], 0:blur_img.shape[1]]

	return crop_output


blur_img = cv2.imread("/Users/sachin007/Documents/EE610/EE610_ImgProcessn/Assgn2/blurred1.png",1) #load the blurred image
cv2.imshow("blurred_image", blur_img)
blur_img = blur_img.astype(float)
Rch,Gch,Bch = cv2.split(blur_img);	#split the image into the respective channels

row_b,cols_b,chan = np.shape(blur_img)
shift_fft_mat = np.ones((2*row_b,2*cols_b))
for i in range(2*row_b):
	for j in range(2*cols_b):
		if((i+j)%2 == 1):
			shift_fft_mat[i,j] = shift_fft_mat[i,j] * (-1)

kernel= cv2.imread("/Users/sachin007/Documents/EE610/EE610_ImgProcessn/Assgn2/kernel1b.png",1) #load the blurring filter 1 
R_ker,G_ker,B_ker = cv2.split(kernel); #split the kernel into the respective channels
kernel = kernel.astype(float)

R_deblur = deblur(Rch,R_ker,shift_fft_mat)
R_deblur = R_deblur/np.max(R_deblur)
G_deblur = deblur(Gch,G_ker,shift_fft_mat)
G_deblur = G_deblur/np.max(G_deblur)
B_deblur = deblur(Bch,B_ker,shift_fft_mat)
B_deblur = B_deblur/np.max(B_deblur)

deblur_img = cv2.merge([R_deblur,G_deblur,B_deblur])
deblur_img = deblur_img.astype(float)
# import pdb;pdb.set_trace()
out_psnr = psnr(blur_img,deblur_img)
print(out_psnr)
cv2.imshow("output",deblur_img)
cv2.waitKey(0)