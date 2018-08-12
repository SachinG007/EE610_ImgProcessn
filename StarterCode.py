import numpy as np 

# img = np.matrix([[1,2,3,4,6],[2,3,4,6,8],[5,6,8,9,3]])
# print(img)

# patch = img[1 - 1:1+1,1-1:1+1]
# print(patch)

# # arr = np.reshape(img,(15,1))
# # print(arr)
# # print(arr.size)

# mean = (0,0)
# cov = [[1, 0], [0, 1]]
# x = np.random.multivariate_normal(mean, cov, (3, 3))
# print(x)
# x.shape()

x = np.arange(-5, 5, 1)
y = np.arange(-5, 5, 1)
xx, yy = np.meshgrid(x, y, sparse=True)
print(xx)