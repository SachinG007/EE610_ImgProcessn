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
import argparse
from skimage.measure import compare_ssim as ssim


parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default="inverse_truncated", choices=["full_inverse", "inverse_truncated","weiner","least_squares"], help="which method u want to use for deblurring")
parser.add_argument("--constant", type=float, default=1,help="percentage of average value u want to give as a constant in weiner and the least least squares")
parser.add_argument("--cutoff", type=int, default=140,help="radius of butterworth")
a = parser.parse_args()


# def ssim(image1, image2):
#     image1 = cv2.normalize(image1, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
#     image2 = cv2.normalize(image2, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
#     return ssim(image1, image2)


def psnr(img1, img2):	#input,output
    mse = np.mean( (img1 - img2) ** 2 ).astype(float)
    inp_max = np.max(img1)
    return 20 * math.log10(inp_max / math.sqrt(mse))
    # return mse

def self_ssim(img1,img2):
        
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    var1 = np.var(img1)
    var2 = np.var(img2)

    cov_img = (img1-mean1)*(img2-mean2)
    cov = np.mean(cov_img) 
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    numerator = (2*mean1*mean2 + c1)*(2*cov + c2)
    denom = (mean1**2 + mean2**2 + c1)*(var1 + var2 + c2)

    return numerator/denom

def butterworth_filter(size, order):

    rows = size[0]
    cols = size[1]
    
    x = np.arange(-cols/2,cols/2 ,1)
    y = np.arange(-rows/2,rows/2 ,1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.sqrt(xx**2 + yy**2)
    # print(z)
    cutoff = a.cutoff
    butterworth = 1/(1 + np.power(z/cutoff, 2*order))
    # cv2.imshow("butterworth",butterworth)
    # pdb.set_trace()
        
    return butterworth

def full_inverse(fft_kernel, fft_blur_img):

    return fft_blur_img/fft_kernel

def inverse_truncated(fft_kernel, fft_blur_img):

    rows,cols = np.shape(fft_blur_img)

    output = fft_blur_img/fft_kernel
    butterwoth = butterworth_filter((rows,cols),10)
    output = output * butterwoth
    return output 

def weiner(fft_kernel, fft_blur_img):

    numerator = (np.abs(fft_kernel))**2
    denom_right = numerator + np.average(numerator) * a.constant
    denom = np.multiply(fft_kernel,denom_right)

    output = fft_blur_img * numerator/denom
    return output

def constrained_ls(fft_kernel, fft_blur_img, fft_p):

    gamma = 100000 * a.constant
    numerator = np.conj(fft_kernel)
    # gamma = np.average(np.abs(fft_kernel)**2) * a.constant
    denom = np.abs(fft_kernel)**2 + gamma * np.abs(fft_p)**2 
    return fft_blur_img*numerator/denom

def deblur(blur_img, kernel):

    p = [[0 , -1 , 0 ],[-1 , 4 ,-1],[0,-1,0]]
    rows, cols = np.shape(blur_img)

    fft_blur_img = np.fft.fftshift(np.fft.fft2(blur_img,(2*rows,2*cols)))
    fft_kernel = np.fft.fftshift(np.fft.fft2(kernel,(2*rows,2*cols)))
    fft_p = np.fft.fftshift(np.fft.fft2(p,(2*rows,2*cols)))

    if (a.method=="weiner"):
        output = weiner(fft_kernel,fft_blur_img)
    elif (a.method == "least_squares"):
        output = constrained_ls(fft_kernel, fft_blur_img, fft_p)
    elif (a.method == "inverse_truncated"):
        output = inverse_truncated(fft_kernel,fft_blur_img)
    elif (a.method == "full_inverse"):
        output = full_inverse(fft_kernel,fft_blur_img)
    
    output = np.fft.ifft2(np.fft.ifftshift(output))
    output = np.real(output)

    # crop_output = output[0:rows, 0:cols]

    return output

gt = cv2.imread("/Users/sachin007/Documents/EE610/EE610_ImgProcessn/Assgn2/Images/gt.jpg",1) #load the 
blur_img = cv2.imread("/Users/sachin007/Documents/EE610/EE610_ImgProcessn/Assgn2/part4.png",1) #load the blurred image
# cv2.imshow("terimaka",blur_img)
blur_img = cv2.resize(blur_img,(280,372))
Bch,Gch,Rch = cv2.split(blur_img);	#split the image into the respective channels


kernel= cv2.imread("/Users/sachin007/Documents/EE610/EE610_ImgProcessn/Assgn2/part4_kernel.png",0) #load the blurring filter 1 
# kernel = cv2.resize(kernel,(25,25))

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
r,c,ch = np.shape(blur_img)
print(r,c)
deblur_img = np.zeros((2*r,2*c,ch)).astype('uint8')
deblur_img[:,:,0] = B_deblur
deblur_img[:,:,1] = G_deblur
deblur_img[:,:,2] = R_deblur

crop_out = np.zeros((r,c,3)).astype('uint8')
r_2 = int(r/2)
c_2 = int(c/2)
r_34 = int(3*2*r/4)
c_34 = int(3*2*c/4)
r_4 = int(2*r/4)
c_4 = int(2*c/4)

for k in range(3):
    # pdb.set_trace()
    crop_out[0:r_2,0:c_2,k] = deblur_img[r_34:,c_34:,k]
    crop_out[0:r_2,c_2:,k] = deblur_img[r_34:,0:c_4,k]
    crop_out[r_2:,0:c_2,k] = deblur_img[0:r_4,c_34:,k]
    crop_out[r_2:,c_2:,k] = deblur_img[0:r_4,0:c_4,k]

crop_out= crop_out + 5

deblur_img2 = cv2.cvtColor(deblur_img,cv2.COLOR_BGR2RGB)
gt = cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)
blur_img = cv2.cvtColor(blur_img,cv2.COLOR_BGR2RGB)
crop_out = cv2.cvtColor(crop_out,cv2.COLOR_BGR2RGB)
# deblur_img2 = deblur_img
# pdb.set_trace()
# out_psnr = psnr(gt,deblur_img2)
# print("psnr calculated",out_psnr)

# # out_ssim = self_ssim(gt,deblur_img)
# # print("ssim calculated",out_ssim)

# np_ssim = ssim(gt,deblur_img2,multichannel='True')
# print("scikit ssim",np_ssim)
# pdb.set_trace()
plt.subplot(121)
plt.imshow(blur_img)
plt.subplot(122)
plt.imshow(crop_out)
# plt.subplot(223)
# plt.imshow(kernel, cmap = 'gray')
plt.show()
# cv2.waitKey(0)