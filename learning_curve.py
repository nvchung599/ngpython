from theta_optimizer import *

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# get data from csv
data = np.genfromtxt('MacdonellDF.csv', delimiter=',')
np.random.shuffle(data)
x = data[:,1]
y = data[:,2]
x = x[~np.isnan(x)]
y = y[~np.isnan(y)]
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
m = np.size(x,0)

percent_list = np.arange(1,100,2)
plot_length = np.size(percent_list)
cost_tr_vec = np.zeros(plot_length)
cost_cv_vec = np.zeros(plot_length)
print(percent_list.shape)
print(cost_cv_vec.shape)
index = 0
for percent in percent_list:
    x_partial = part_data(x, percent)
    y_partial = part_data(y, percent)

    # construct feature matrix, normalize, bias
    X = mod_degree(x_partial, 5)
    X = normalize(X)
    X = add_bias(X)

    #split data
    tr_X, cv_X, te_X = split_data(X)
    tr_y, cv_y, te_y = split_data(y_partial)

    # train theta
    reg_const = 1
    my_opter = ThetaOptimizer(100, 0.000001, 0.1)
    theta_tr, cost_tr_regged = my_opter.optimize_theta(tr_X, tr_y, reg_const)

    cost_tr = calc_cost(tr_X, tr_y, theta_tr, 0)
    cost_cv = calc_cost(cv_X, cv_y, theta_tr, 0)
    cost_tr_vec[index] = cost_tr
    cost_cv_vec[index] = cost_cv
    index = index + 1

plt.title('Learning Curve')
plt.xlabel('percent of sample size')
plt.ylabel('optimized cost')
plt.plot(percent_list, cost_tr_vec, marker='+', label='TR cost')
plt.plot(percent_list, cost_cv_vec, marker='o', label='CV cost')
plt.legend()
plt.show()