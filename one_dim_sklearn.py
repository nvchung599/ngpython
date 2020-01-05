import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random


def add_bias_to_vector(x_data):

    length = x_data.size
    #length = x_data.size(0)
    ones_vec = np.ones((length, 1))

    new_matrix = np.hstack((ones_vec, x_data))

    return new_matrix

def plot_lin_func(intercept, coef_vec, x_data):
    theta_vec = np.vstack((intercept, coef_vec))
    X = add_bias_to_vector(x_data)
    h_vec = np.matmul(X, theta_vec)
    plt.plot(x_data, h_vec, 'r')


df = pd.read_csv('Consumption.csv')

mu = df.iloc[:,2].mean()
sigma = mu * 0.1
noise = np.random.normal(mu * 0.1, sigma, 200)
df.iloc[:,2] = df.iloc[:,2] + noise

x_data = df.iloc[:,1]
y_data = df.iloc[:,2]
x_data = x_data.values.reshape(-1,1)
y_data = y_data.values.reshape(-1,1)

#print(x_data.shape)
#print(x_data[0:10:1, :])

#my_reg = linear_model.LinearRegression()
my_reg = linear_model.Ridge(alpha = 100000000000000000.0)
my_reg.fit(x_data, y_data)
#print(my_reg.predict([[50000], [100000], [150000]]))
print('coeff')
print(my_reg.coef_)
print('intercept')
print(my_reg.intercept_)
print('plot_lin_func vvv')
plot_lin_func(my_reg.intercept_, my_reg.coef_, x_data)


plt.scatter(x_data, y_data)
plt.show()

#x, y = parse_data('1.01. Simple linear regression.csv')
#data = np.genfromtxt('MacdonellDF.csv', delimiter=',')
#np.random.shuffle(data)
#x = data[:,1]
#y = data[:,2]
#x = x[~np.isnan(x)]
#y = y[~np.isnan(y)]
#x = x.reshape(-1, 1)
#y = y.reshape(-1, 1)