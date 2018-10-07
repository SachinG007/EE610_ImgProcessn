import numpy as np
import cv2

def inverse_filter(blurred_image, kernel):
	rows, cols = blurred_image.shape
	pad_blurred_image = np.zeros((2*rows, 2*cols), np.float64)
	pad_blurred_image[0:blurred_image.shape[0], 0:blurred_image.shape[1]] = blurred_image
	
	for i in range(pad_blurred_image.shape[0]):
		for j in range(pad_blurred_image.shape[1]):
			if (i+j) % 2 != 0:
				pad_blurred_image[i,j] = pad_blurred_image[i,j]*(-1)

	FFT_pad_blurred_image = np.fft.fft2(pad_blurred_image)
	
	FFT_kernel = np.fft.fft2(kernel)
	recovered_channel = np.fft.ifft2(np.divide(FFT_pad_blurred_image, np.abs(FFT_kernel))).real
	recovered_channel = recovered_channel.astype(np.float64)

	for i in range(recovered_channel.shape[0]):
		for j in range(recovered_channel.shape[1]):
			if (i+j) % 2 != 0:
				recovered_channel[i,j] = (-1)*recovered_channel[i,j]

	result = recovered_channel[0:rows, 0:cols]
	print(result.dtype)
	return result

if __name__ == '__main__':
	blurred_image = cv2.imread("Blurry1_1.jpg", 1)
	cv2.imshow("blurred_image", blurred_image)

	R_ch, G_ch, B_ch = cv2.split(blurred_image)
	blurred_image = blurred_image.astype(np.float64)

	kernel = cv2.imread("kernel_hai.png", 1)
	# kernel = cv2.resize(kernel, (1600,1600,3))
	kernel_R, kernel_G, kernel_B = cv2.split(kernel)

	kernel_R = cv2.resize(kernel_R, (1600,1600)).astype(np.float64)
	kernel_G = cv2.resize(kernel_G, (1600,1600)).astype(np.float64)
	kernel_B = cv2.resize(kernel_B, (1600,1600)).astype(np.float64)
	
	rec_R_ch = inverse_filter(R_ch, kernel_R)
	rec_R_ch = rec_R_ch/np.max(rec_R_ch)

	rec_G_ch = inverse_filter(G_ch, kernel_G)
	rec_G_ch = rec_G_ch/np.max(rec_G_ch)
	
	rec_B_ch = inverse_filter(B_ch, kernel_B)
	rec_B_ch = rec_B_ch/np.max(rec_B_ch)
	
	recovered_image = cv2.merge([rec_R_ch, rec_G_ch, rec_B_ch])
	cv2.imshow("recovered_image", recovered_image)
	cv2.waitKey(0)