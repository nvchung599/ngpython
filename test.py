import numpy as np
from matplotlib import pyplot as plt
from general import *

toggle = 1

if toggle == 0:
    init_nested_list(5,3)

if toggle == 1:
    x, y = parse_data('1.01. Simple linear regression.csv')
    x_train, y_train, x_test, y_test, x_cv, y_cv = split_data(x, y)

    deg_vec = np.array([1, 2])
    reg_vec = np.array([0.1, 1, 10])

    deg_qty = np.size(deg_vec, 0)
    reg_qty = np.size(reg_vec, 0)

    theta_mat = init_nested_list(deg_qty, reg_qty)
    J_mat = np.zeros((deg_qty, reg_qty))

    for i in range(0, deg_qty):
        for j in range(0, reg_qty):
            X = mod_degree(x_test, i+1)
            theta = construct_theta(X)


if toggle == 2:

    #TODO THIS ALL BELONGS IN MAIN OR SPLIT DATA

    #get data from csv
    data = np.genfromtxt('1.01. Simple linear regression.csv', delimiter=',')
    x = data[:,0]
    y = data[:,1]

    #cleanup data
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)



    #TODO REPLACE WITH MOD POWER
    #construct feature matrix, normalize, bias
    X = np.hstack((x,np.power(x,2),np.power(x,0.5)))

    #TODO MAIN
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








