import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#df = pd.read_csv('1.01. Simple linear regression.csv')
df = pd.read_csv('Consumption.csv')
x_data = df.iloc[:,1]
y_data = df.iloc[:,2]
#x_data = x_data.reshape(-1,1)
#y_data = y_data.reshape(-1,1)
x_data = x_data.values.reshape(-1,1)
y_data = y_data.values.reshape(-1,1)

#plt.scatter(df.iloc[:,0], df.iloc[:,1])
#plt.show()

my_reg = linear_model.LinearRegression()
my_reg.fit(x_data, y_data)
print(my_reg.predict([[50000], [100000], [150000]]))

#x, y = parse_data('1.01. Simple linear regression.csv')
#data = np.genfromtxt('MacdonellDF.csv', delimiter=',')
#np.random.shuffle(data)
#x = data[:,1]
#y = data[:,2]
#x = x[~np.isnan(x)]
#y = y[~np.isnan(y)]
#x = x.reshape(-1, 1)
#y = y.reshape(-1, 1)