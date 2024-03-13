import numpy as np

a = np.floor(10*np.random.random((3,4)))
print("3행 4열 배열 생성\n", a)
print("a 배열의 shape = ",a.shape)

print("a.ravel 결과값\n",a.ravel())  # returns the array, flattened
print("a.reshape(6,2) 결과값\n",a.reshape(6,2))  # returns the array with a modified shape
print("a.T 결과값\n",a.T)  # returns the array, transposed
print("a.T.shape 결과값 = ",a.T.shape)
print("a.shape 결과값 = ",a.shape)