import numpy as np
from matplotlib import pyplot as plt

def parse_data(path):
    """extracts csv data and cleans it up"""
    data = np.genfromtxt(path, delimiter=',')
    x = data[:,0]
    y = data[:,1]
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    return (x,y)

def split_data(x, y):
    """splits x and y data 60/20/20. accepts arbitrary dataset sizes."""
    m = np.size(x, 0)
    partition_one = int(round(m*0.6))
    partition_two = int(round(m*0.8))
    x_train, x_test, x_cv = x[:partition_one,:], x[partition_one:partition_two,:], x[partition_two:,:]
    y_train, y_test, y_cv = y[:partition_one,:], y[partition_one:partition_two,:], y[partition_two:,:]
    return x_train, y_train, x_test, y_test, x_cv, y_cv

def init_nested_list(rows, cols):
    """initializes an empty (None) nested list/matrix. To be populated with
    vectors of varying lengths (theta vectors)
    note: to access... 2dlist[row][col]"""
    out_list = [None] * rows
    in_list = [None] * cols
    for i in range(0, rows):
        out_list[i] = in_list
    return out_list

def mod_degree(x, deg):
    """if degree is specified to be >1, creates additional features and
    constructs the matrix X"""
    X = np.copy(x)
    if deg > 1:
        for i in range(2, deg+1):
            add_me = np.copy(x)**i
            X = np.hstack([X, add_me])
    return X

def init_nested_list_data(x, max_deg):
    """returns a list populated with matrices of increasing degree based on x"""
    deg_feat_list = [None] * max_deg
    for i in range(0, max_deg):
        deg_feat_list[i] = mod_degree(x, i+1)
    return deg_feat_list


def construct_theta(X):
    """random inits a theta vector of compatible size with X.
    call this AFTER bias has been added to X"""
    n_plus_one = np.size(X,1)
    theta = np.random.rand(n_plus_one, 1)
    return theta

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

def calc_cost(X, y, theta, reg_const):
    """Calculates cost across all m examples of a dataset"""
    m = np.size(X,0)
    hypo = np.matmul(X, theta) # (m*n)*(n*1) = m*1
    err = hypo - y
    sqr_err = np.power(err, 2)
    sum_sqr_err = np.sum(sqr_err)

    temp_theta = np.copy(theta)
    temp_theta[0] = 0
    temp_theta = temp_theta**2
    reg_term = reg_const*np.sum(temp_theta)

    cost = (sum_sqr_err + reg_term)/(2*m)
    return cost

def calc_grad(X, y, theta, reg_const):
    """Calculates cost across all m examples of a dataset"""
    m = np.size(X,0)
    hypo = np.matmul(X, theta) # (m*n)*(n*1) = m*1
    err = hypo - y # m*1
    accum_term = np.matmul(np.transpose(X), err) # (4*m)*(m*1) = 4*1
    grad = (accum_term + reg_const*theta)/m
    return grad

def grad_check(X, y, theta, epsilon, reg_const):
    """numerically calculates parameter gradients for the first learning step.
    prints side by side with matrix-calculated gradients for verification"""
    n = np.size(theta,0)
    ep_mat = np.identity(n)*epsilon
    mat_grad = calc_grad(X, y, theta, reg_const)
    num_grad = np.zeros((n,1))
    for i in range(0,n):
        ep_vec = ep_mat[:,[i]]
        J_hi = calc_cost(X, y, theta+ep_vec, reg_const)
        J_lo = calc_cost(X, y, theta-ep_vec, reg_const)
        num_grad[i,0] = (J_hi-J_lo)/(2*epsilon)
    print(mat_grad)
    print(num_grad)

    




