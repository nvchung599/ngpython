from theta_optimizer import *

toggle = 1

if toggle == 0:
    foo = "bar"

if toggle == 1:
    x, y = parse_data('1.01. Simple linear regression.csv')

    max_deg = 3
    deg_data_list = gen_degree_cases(x, max_deg)
    for i in range(0, len(deg_data_list)):
        deg_data_list[i] = normalize(deg_data_list[i])
        deg_data_list[i] = add_bias(deg_data_list[i])
    deg_set_data_nlist = split_into_nlist(deg_data_list)
    #TODO SPLIT Y DATA INTO LIST
    tr_label, cv_label, te_label = split_data(y)
    set_label_list = [tr_label, cv_label, te_label]

    reg_const_list = [0.1, 1, 10]
    deg_reg_theta_nlist = init_nested_list(len(deg_data_list), len(reg_const_list))
    deg_reg_Jtr_nlist = init_nested_list(len(deg_data_list), len(reg_const_list))
    deg_reg_Jcv_nlist = init_nested_list(len(deg_data_list), len(reg_const_list))
    deg_reg_Jte_nlist = init_nested_list(len(deg_data_list), len(reg_const_list))

    #deg_set_data_nlist      2x3
    #deg_reg_theta_nlist     2x2

    my_opter = ThetaOptimizer(100, 0.000001, 0.1)

    for i in range(0, len(deg_data_list)):
        for j in range(0, len(reg_const_list)):
            cur_X = deg_set_data_nlist[i][0]
            cur_y = set_label_list[0]
            cur_reg_const = reg_const_list[j]
            cur_theta, cur_cost = my_opter.optimize_theta(cur_X, cur_y, cur_reg_const)
            deg_reg_theta_nlist[i][j] = cur_theta
            deg_reg_Jtr_nlist[i][j] = cur_cost


    i_min = None
    j_min = None
    cost_min = 9999
    for i in range(0, len(deg_data_list)):
        for j in range(0, len(reg_const_list)):
            cur_X = deg_set_data_nlist[i][1]
            cur_y = set_label_list[1]
            cur_reg_const = reg_const_list[j]
            cur_theta = deg_reg_theta_nlist[i][j]
            cur_cost = calc_cost(cur_X, cur_y, cur_theta, cur_reg_const)
            deg_reg_Jcv_nlist[i][j] = cur_cost
            if cur_cost < cost_min:
                cost_min = cur_cost
                i_min = i
                j_min = j


    print(deg_reg_Jtr_nlist)
    print(deg_reg_Jcv_nlist)

    print('optimal deg-reg combination')
    print('index i=%i, index j=%i' % (i_min, j_min))
    print('degree=%i, reg_const=%d' % (i_min+1, reg_const_list[j_min]))
    print('cost_min=%d' % cost_min)




# SINGLE DEGREE/LAMBDA OPTIMIZATION
if toggle == 2:
    # get data from csv
    x, y = parse_data('1.01. Simple linear regression.csv')
    # construct feature matrix, normalize, bias
    X = mod_degree(x, 5)
    X = normalize(X)
    X = add_bias(X)
    # train theta
    my_opter = ThetaOptimizer(100, 0.000001, 0.1)
    theta, cost = my_opter.optimize_theta(X, y, 1)
    my_opter.plot_last()







