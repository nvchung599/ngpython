import numpy as np
from matplotlib import pyplot as plt


def normalize(X):
    """normalizes every feature column in the X matrix wrt mean and stdev
    note: apply this function BEFORE adding a ones column
    note: shape of matrix should be m*n"""
    n = np.size(X,1)
    X_norm = np.copy(X)
    for i in range(0, n):
        feature_vec = X_norm[:,[i]]
        my_std = np.std(feature_vec)
        my_mean = np.mean(feature_vec)
        X_norm[:,[i]] = X_norm[:,[i]] - my_mean
        X_norm[:,[i]] = np.divide(X_norm[:,[i]],my_std)
    return X_norm


def add_bias(X):
    """Adds a column of ones on the left side of a 2D matrix
    note: a numpy matrix must be shape (m,1) NOT (m,)"""
    m = np.size(X,0)
    ones = np.ones([m,1])
    X_bias = np.hstack([ones, X])
    return X_bias

def calc_cost(X, y, theta):
    """Calculates cost across all m examples of a dataset"""
    m = np.size(X,0)
    hypo = np.matmul(X, theta) # (m*n)*(n*1) = m*1
    err = hypo - y
    sqr_err = np.power(err, 2)
    cost = (np.sum(sqr_err))/(2*m)
    return cost

def calc_grad(X, y, theta):
    """Calculates cost across all m examples of a dataset"""
    m = np.size(X,0)
    hypo = np.matmul(X, theta) # (m*n)*(n*1) = m*1
    err = hypo - y # m*1
    accum_term = np.matmul(np.transpose(X), err) # (4*m)*(m*1) = 4*1
    grad = accum_term/m
    return grad

def grad_check(X, y, theta, epsilon):
    n = np.size(theta,0)
    ep_mat = np.identity(n)*epsilon
    mat_grad = calc_grad(X, y, theta)
    num_grad = np.zeros((n,1))
    for i in range(0,n):
        ep_vec = ep_mat[:,[i]]
        J_hi = calc_cost(X, y, theta+ep_vec)
        J_lo = calc_cost(X, y, theta-ep_vec)
        num_grad[i,0] = (J_hi-J_lo)/(2*epsilon)
    print(mat_grad)
    print(num_grad)

    




