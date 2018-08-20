# a = 0
# b = 10

# def f():
# 	global a,b
# 	b = a
# 	a = 20

# def g():
# 	global a,b
# 	a = b

# f()
# g()
# print(a,b)
import numpy as np
A = np.arange(8).reshape((4,2))
print(A)

A = np.flip(A,0)
A = np.flip(A,1)
print(A)
print(np.sum(A))