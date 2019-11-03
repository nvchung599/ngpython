import numpy as np
from matplotlib import pyplot as plt
from general import *

toggle_one = 0
toggle_two = 1

if toggle_one == 1:
    x = np.array([[1,2,3,4,5,6,7,8,9]])
    x = x.reshape(-1,1)
    y = np.copy(x)-1
    print(x)
    print(np.transpose(x))

if toggle_two == 1:

    #get data from csv
    data = np.genfromtxt('1.01. Simple linear regression.csv', delimiter=',')
    x = data[:,0]
    y = data[:,1]

    #cleanup data
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)

    #construct feature matrix, normalize, bias
    X = np.hstack((x,np.power(x,2),np.power(x,0.5)))
    X = normalize(X)
    X = add_bias(X)
    print(X.shape)

    theta = np.random.rand(4,1)
    print(theta.shape)
    
    cost = calc_cost(X, y, theta)
    grad = calc_grad(X, y, theta)

    print(cost)
    print(theta)
    print(grad)

    #TODO GRADIENT CHECKING











