import numpy as np
from matplotlib import pyplot as plt
import tkinter

data = np.genfromtxt('MacdonellDF.csv', delimiter=',')

np.random.shuffle(data)

x = data[:,1]
y = data[:,2]
x = x[~np.isnan(x)]
y = y[~np.isnan(y)]
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

print(x.shape)
print(y.shape)

print(data)
plt.title('demo')
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.scatter(x, y)
plt.show()



#a = np.array([1,2,3])
#b = np.array([[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8]])
#
#print(b[0, 2:-1:2])
#
#b[0,:] = 5
#print(b)
#
#print('\n\n\n')
#c = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
#
#print(np.full((2,3,4), 99))
#
#print(np.full_like(c,100))
#
#print(np.random.randint(10, 100, size=(3,3 )))
#
#print(np.identity(6))
#
#d = np.full((2,3),5)
#e = np.full((3,1),5)
#print(np.matmul(d,e))
#
#print(np.sum(c))
#
#identity = np.identity(5)
#bias = np.ones([1,5])
#print(np.vstack([bias, identity]))
#x = np.vstack([bias, identity])
#print(np.size(x,0))
#newones = np.ones([np.size(x,0),1])
#y = np.hstack([newones, x])
#
#print(y[y == 1])
#
#
