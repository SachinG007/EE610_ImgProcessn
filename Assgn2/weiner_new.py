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


def psnr(img1, img2):	#input,output
    mse = np.mean( (img1 - img2) ** 2 ).astype(float)
    inp_max = np.max(img1)
    return 20 * math.log10(inp_max / math.sqrt(mse))
    # return mse

def constrained_ls(fft_kernel, fft_blur_img, fft_p):

    gamma = 100000
    numerator = np.conj(fft_kernel)
    denom = np.abs(fft_kernel)**2 + gamma * np.abs(fft_p)**2

    return fft_blur_img*numerator/denom

def weiner(fft_kernel, fft_blur_img):

    numerator = (np.abs(fft_kernel))**2
    denom_right = numerator + np.average(numerator)
    denom = np.multiply(fft_kernel,denom_right)

    output = fft_blur_img * numerator/denom
    return output


def deblur(blur_img, kernel):

    p = [[0 , -1 , 0 ],[-1 , 4 ,-1],[0,-1,0]]
    rows, cols = np.shape(blur_img)

    fft_blur_img = np.fft.fftshift(np.fft.fft2(blur_img,(2*rows,2*cols)))
    fft_kernel = np.fft.fftshift(np.fft.fft2(kernel,(2*rows,2*cols)))
    fft_p = np.fft.fftshift(np.fft.fft2(p,(2*rows,2*cols)))


    output = constrained_ls(fft_kernel, fft_blur_img, fft_p)
    # output = weiner(fft_kernel,fft_blur_img)
    output = np.fft.ifft2(np.fft.ifftshift(output))
    output = np.real(output)

    crop_output = output[0:rows, 0:cols]

    return crop_output


blur_img = cv2.imread("/Users/sachin007/Documents/EE610/EE610_ImgProcessn/Assgn2/blurred1.png",1) #load the blurred image
# cv2.imshow("terimaka",blur_img)
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