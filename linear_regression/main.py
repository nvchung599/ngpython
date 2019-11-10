from theta_optimizer import *

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

#x, y = parse_data('1.01. Simple linear regression.csv')
data = np.genfromtxt('MacdonellDF.csv', delimiter=',')
np.random.shuffle(data)
x = data[:,1]
y = data[:,2]
x = x[~np.isnan(x)]
y = y[~np.isnan(y)]
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

max_deg = 5
deg_data_list = gen_degree_cases(x, max_deg)
for i in range(0, len(deg_data_list)):
    deg_data_list[i] = normalize(deg_data_list[i])
    deg_data_list[i] = add_bias(deg_data_list[i])
deg_set_data_nlist = split_into_nlist(deg_data_list)

tr_label, cv_label, te_label = split_data(y)
set_label_list = [tr_label, cv_label, te_label]

#reg_const_list = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
reg_const_list = []
for i in range(0,100):
    reg_const_list.append(i)


deg_reg_theta_nlist = init_nested_list(len(deg_data_list), len(reg_const_list))
deg_reg_Jtr_nlist = init_nested_list(len(deg_data_list), len(reg_const_list))
deg_reg_Jcv_nlist = init_nested_list(len(deg_data_list), len(reg_const_list))
deg_reg_Jte_nlist = init_nested_list(len(deg_data_list), len(reg_const_list))

my_opter = ThetaOptimizer(100, 0.000001, 0.1)

for i in range(0, len(deg_data_list)):
    for j in range(0, len(reg_const_list)):
        cur_X = deg_set_data_nlist[i][0]
        cur_y = set_label_list[0]
        cur_reg_const = reg_const_list[j]
        cur_theta, cur_cost = my_opter.optimize_theta(cur_X, cur_y, cur_reg_const)
        deg_reg_theta_nlist[i][j] = cur_theta

        #compare against cv cost
        deg_reg_Jtr_nlist[i][j] = calc_cost(cur_X, cur_y, cur_theta, 0)


i_min = None
j_min = None
cost_min = 9999
for i in range(0, len(deg_data_list)):
    for j in range(0, len(reg_const_list)):
        cur_X = deg_set_data_nlist[i][1]
        cur_y = set_label_list[1]
        cur_reg_const = reg_const_list[j]
        cur_theta = deg_reg_theta_nlist[i][j]
        cur_cost = calc_cost(cur_X, cur_y, cur_theta, 0)
        deg_reg_Jcv_nlist[i][j] = cur_cost
        if cur_cost < cost_min:
            cost_min = cur_cost
            i_min = i
            j_min = j

#print(deg_reg_Jtr_nlist)
#print(deg_reg_Jcv_nlist)


print('optimal deg-reg combination')
print('index i=%i, index j=%i' % (i_min, j_min))
print('degree=%i, reg_const=%f' % (i_min+1, reg_const_list[j_min]))
print('cost_min=%f' % cost_min)

deg_list = []
for i in range(1, max_deg+1):
    deg_list.append(i)

if 1 == 1:
    x, y, z = map_scatter(deg_list, reg_const_list, deg_reg_Jcv_nlist)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, marker='o', label='CV Cost')
    ax.set_xlabel('degree')
    ax.set_ylabel('lambda')
    ax.set_zlabel('cv cost')
    x, y, z = map_scatter(deg_list, reg_const_list, deg_reg_Jtr_nlist)
    ax.scatter(x, y, z, marker='x', label='TR Cost')
    plt.legend()
    plt.show()

if 1 == 0:
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('my title')
    plt.scatter(x, y)

    x_min = np.min(x)
    x_max = np.max(x)
    x_plotvec = np.linspace(x_min, x_max, 100)
    x_plotvec = x_plotvec.reshape(-1,1)

    theta_opt = deg_reg_theta_nlist[i_min][j_min]


    hypo_plotvec = calc_hypo(x_plotvec, theta_opt)

    plt.plot(x_plotvec, hypo_plotvec)
    plt.show()
