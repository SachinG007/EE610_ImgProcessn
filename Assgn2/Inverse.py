import numpy as np # Import numpy for mathematical operations and functions
import sys # Required for starting and exiting application
import copy # Need the deepcopy function to copy entire arrays
from PIL import Image # Required to read jpeg images
import cv2 as cv2
import math
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import butter, lfilter, freqz
import pdb;

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

def lowpassfilter(size, cutoff, n):

    rows = size[0]
    cols = size[1]
    
    x =  (np.ones((rows,1)) * np.arange(1,cols+1) - (np.fix(cols/2)+1))/cols
#     print(x)
    y =  (np.arange(1,rows+1).reshape(rows,1) * np.ones((1,cols)) - (np.fix(rows/2)+1))/rows
#     print(y)
    
    radius = np.sqrt(np.power(x,2) + np.power(y,2))
    f = 1 / (1.0 + np.power((radius / cutoff),(2*n)))
    # plt.imshow(f,cmap='gray')
    return f


def deblur(blur_img, kernel):


	rows,cols= np.shape(blur_img)

	#taking the respective DFTs
	fft_blur_img = np.fft.fftshift(np.fft.fft2(blur_img,(2*rows,2*cols)))
	fft_kernel = np.fft.fftshift(np.fft.fft2(kernel,(2*rows,2*cols)))

	output = fft_blur_img/fft_kernel
	butterwoth = lowpassfilter((2*rows,2*cols),.1,10)
	output = output * butterwoth


	output = np.fft.ifft2(np.fft.ifftshift(output))
	output = np.real(output)

	crop_output = output[0:rows, 0:cols]

	return crop_output


blur_img = cv2.imread("/Users/sachin007/Documents/EE610/EE610_ImgProcessn/Assgn2/blurred1.png",1) #load the blurred image
Bch,Gch,Rch = cv2.split(blur_img);	#split the image into the respective channels


kernel= cv2.imread("/Users/sachin007/Documents/EE610/EE610_ImgProcessn/Assgn2/cho11.png",0) #load the blurring filter 1 

R_deblur = deblur(Rch,kernel)
R_deblur = (R_deblur/np.max(R_deblur) * 255)
np.clip(R_deblur,0,255,out=R_deblur)
# R_deblur = (R_deblur).astype('uint8')

G_deblur = deblur(Gch,kernel)
G_deblur = (G_deblur/np.max(G_deblur) * 255)
np.clip(G_deblur,0,255,out=G_deblur)
# G_deblur = (G_deblur).astype('uint8')

B_deblur = deblur(Bch,kernel)
B_deblur = (B_deblur/np.max(B_deblur) * 255)
np.clip(B_deblur,0,255,out=B_deblur)
# B_deblur = (B_deblur).astype('uint8')
# pdb.set_trace()

deblur_img = np.zeros((np.shape(blur_img))).astype('uint8')
deblur_img[:,:,0] = B_deblur
deblur_img[:,:,1] = G_deblur
deblur_img[:,:,2] = R_deblur

deblur_img2 = cv2.cvtColor(deblur_img,cv2.COLOR_BGR2RGB)
out_psnr = psnr(blur_img,deblur_img2)
print(out_psnr)
plt.imshow(deblur_img2)
plt.show()
cv2.waitKey(0)