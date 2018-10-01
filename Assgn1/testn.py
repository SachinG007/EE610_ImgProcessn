
def f():
	global a,b
	a = 5
	b = a

def g():
	global a
	a = 78

def h():

	global a,b
	a = b


f()
g()
g()
h()
print(a,b)
# import numpy as np
# A = np.arange(8).reshape((4,2))
# print(A)

# A = np.flip(A,0)
# A = np.flip(A,1)
# print(A)
# print(np.sum(A))