import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


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

def plot_lin_func(intercept, coef_vec, x_data, best_degree):
    # get approx range of input x
    # arange a new x vector
    # do as below
    min = np.min(x_data[:,0])
    max = np.max(x_data[:,0])
    print(min)
    print(max)
#    x_space = np.linspace(min, max, 200)
    print(intercept.shape)
    print(coef_vec.shape)
    theta_vec = np.vstack((intercept, coef_vec))
    X = up_degree(x_data, best_degree)
    X = add_bias_to_matrix(X)
    h_vec = np.matmul(X, theta_vec)
    plt.plot(x_data[:,0], h_vec, 'r')


df = pd.read_csv('Consumption.csv')

mu = df.iloc[:,2].mean()
sigma = mu * 0.1
noise = np.random.normal(mu * 0.1, sigma, 200)
df.iloc[:,2] = df.iloc[:,2] + noise

x_data = df.iloc[:,1]
y_data = df.iloc[:,2]
x_data = x_data.values.reshape(-1,1)
y_data = y_data.values.reshape(-1,1)

my_degrees = np.arange(2,54)
r2_scores = []

best_r2 = 0
best_intercept = 0
best_coef = None
best_degree = 0

for degree in my_degrees:

    x_enhanced = up_degree(x_data, degree)

    x_train, x_test, y_train, y_test = train_test_split(x_enhanced, y_data, test_size=0.33, random_state=42)

    my_reg = linear_model.LinearRegression(normalize=True)
    my_reg.fit(x_train, y_train)

    y_pred = my_reg.predict(x_test)
#    print('degree = %i' % degree)
#    print('mean square error = %i' % mean_squared_error(y_test, y_pred))
#    print('r2 score = %f' % r2_score(y_test, y_pred))

    r2_scores.append(r2_score(y_test, y_pred))

    if best_r2 < r2_score(y_test, y_pred):
        best_r2 = r2_score(y_test, y_pred)
        best_coef = my_reg.coef_
        best_intercept = my_reg.intercept_
        best_degree = degree


toggle = 1

if toggle == 0:
    plt.plot(my_degrees, r2_scores)
    plt.show()

if toggle == 1:
    plot_lin_func(best_intercept.reshape(-1,1), np.transpose(best_coef), x_data, best_degree)
    plt.scatter(x_data, y_data)
    print('best_degree = %i' % best_degree)
    print('best_r2score = %f' % best_r2)
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