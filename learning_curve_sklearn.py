import numpy as np
from numpy import random
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import random

def add_noise(y_data):
    return 1
    #gen some random noise factor close to 1
    #mult y_data by factor
    # return y

def up_degree(x_data, degree):
    X = np.copy(x_data)
    for deg in range(degree-1):
        current_power = deg + 2
        stack_me = np.power(x_data, current_power)
        stack_me = stack_me.reshape(-1,1)
        X = np.hstack((X, stack_me))
    return X

def add_bias_to_matrix(x_data):
    length = np.size(x_data, 0)
    ones_vec = np.ones((length, 1))
    new_matrix = np.hstack((ones_vec, x_data))
    return new_matrix

def plot_lin_func(intercept, coef_vec, x_data):
    # get approx range of input x
    # arange a new x vector
    # do as below
    theta_vec = np.vstack((intercept, coef_vec))
    X = add_bias_to_matrix(x_data)
    h_vec = np.matmul(X, theta_vec)
    plt.plot(x_data[:,0], h_vec, 'r')


df = pd.read_csv('Consumption.csv')

mu = df.iloc[:,2].mean()
sigma = mu * 0.1
noise = np.random.normal(mu * 0.1, sigma, 200)
df.iloc[:,2] = df.iloc[:,2] + noise
df.iloc[:,2] = df.iloc[:,2] + noise

m_vec = np.arange(40,200)
r2_scores_train = []
r2_scores_test = []

for m in m_vec:

    x_data = df.iloc[0:m:1,1]
    y_data = df.iloc[0:m:1,2]
    x_data = x_data.values.reshape(-1,1)
    y_data = y_data.values.reshape(-1,1)

    degree = 5

    x_enhanced = up_degree(x_data, degree)

    x_train, x_test, y_train, y_test = train_test_split(x_enhanced, y_data, test_size=0.33, random_state=42)

    my_reg = linear_model.LinearRegression(normalize=True)
    my_reg.fit(x_train, y_train)

    y_pred_train = my_reg.predict(x_train)
    y_pred_test = my_reg.predict(x_test)
    print('degree = %i' % degree)
    print('mean square error = %i' % mean_squared_error(y_test, y_pred_test))
    print('r2 score = %f' % r2_score(y_test, y_pred_test))

    r2_scores_train.append(r2_score(y_train, y_pred_train))
    r2_scores_test.append(r2_score(y_test, y_pred_test))

plt.plot(m_vec, r2_scores_train)
plt.plot(m_vec, r2_scores_test)
plt.legend(['train', 'test'])
plt.show()


#plot_lin_func(my_reg.intercept_, np.transpose(my_reg.coef_), x_train)
#plt.scatter(x_train[:,0], y_train)
#plt.show()

#x, y = parse_data('1.01. Simple linear regression.csv')
#data = np.genfromtxt('MacdonellDF.csv', delimiter=',')
#np.random.shuffle(data)
#x = data[:,1]
#y = data[:,2]
#x = x[~np.isnan(x)]
#y = y[~np.isnan(y)]
#x = x.reshape(-1, 1)
#y = y.reshape(-1, 1)