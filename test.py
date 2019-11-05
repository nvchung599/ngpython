import numpy as np
from matplotlib import pyplot as plt
from general import *

toggle = 2

if toggle == 1:
    x = np.ones((5,1))
    y = np.ones((5,1))
    z = np.hstack([x, y])
    print(z)


if toggle == 2:

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

    theta = np.random.rand(4,1)
    
    it = 0
    it_max = 100
    dJ_current = 999
    dJ_stable = 0.000001
    J_history = []
    alpha = 0.1
    reg_const = 0
    
    while it<it_max and dJ_current>dJ_stable: 
        J_history = np.append(J_history, calc_cost(X, y, theta, reg_const))
        grad = calc_grad(X, y, theta, reg_const)
        if it>0:
            dJ_current = J_history[it-1] - J_history[it]
        it = it+1
        theta = theta-alpha*grad
    print('theta_optimized:')
    print(theta)
    print('J_history:')
    print(J_history)


    #gradient checking with toggle
    if 1 == 1:
        epsilon = 0.01
        grad_check(X, y, theta, epsilon, reg_const)

    plt.title('J History')
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.plot(J_history)
    plt.show()

    #TODO add regularization functions, 








