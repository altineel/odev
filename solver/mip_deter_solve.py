import numpy as np
import gurobipy as gp
from common import *
from sim_environ.cost_functions import fuel_price_func, fuel_consume_const_func, fixed_bunkering_costs
import json
import pickle as pkl


def solve_mip_deter(dist_mat=DIST_MAT,
                    fuel_consume_rate=DYN_VEL_FUEL_CONSUM_CONST,
                    fuel_price_list=EXPECTED_BUNKERING_COSTS,
                    fuel_capacity=FUEL_CAPACITY,
                    sched=ROUTE_SCHEDULE,
                    filename_lp=FILENAME_LP,
                    filename_dm=FILENAME_DM,
                    file_out_name="model_lp_mari_deter.lp"):
    # GRB = gp.GRB
    n = dist_mat.shape[0]

    m = gp.Model("deter_v2")
    x = {}
    x2_coef_list = []
    x1_coef_list = []
    b_coef_list = []

    len_ = len(sched) - 1
    for n in range(len_):
        travel_dist_n_next = dist_mat[sched[n]][sched[n + 1]]

        p_n = fuel_price_func(sched[n], fuel_price_list=fuel_price_list)
        p_n_next = fuel_price_func(sched[n + 1])

        # fuel_consume_n_next = fuel_consume_const_func(travel_dist_n_next)
        fuel_cons_rate = DYN_VEL_FUEL_CONSUM_CONST*np.power(REGULAR_SPEED, 2)

        fuel_consume_n_next = (travel_dist_n_next/REGULAR_SPEED)*fuel_cons_rate

        x2_coef = p_n
        x1_coef = -p_n  # + p_n_next
        b_coef = fixed_bunkering_costs(n)

        x2_coef_list.append(x2_coef)
        x1_coef_list.append(x1_coef)
        b_coef_list.append(b_coef)

        var_x2_key = "var_x2.." + str(sched[n])
        var_x2_key_next = "var_x2.." + str(sched[n] + 1)
        var_x1_key = "var_x1.." + str(sched[n])
        var_x1_next_key = "var_x1.." + str(sched[n] + 1)
        var_b_key = "var_b.." + str(sched[n])

        x[var_x2_key] = m.addVar(name=var_x2_key, vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=FUEL_CAPACITY)
        if var_x1_key not in x:
            x[var_x1_key] = m.addVar(name=var_x1_key, vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=FUEL_CAPACITY)
        if var_x1_next_key not in x:
            x[var_x1_next_key] = m.addVar(name=var_x1_next_key, vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=FUEL_CAPACITY)
        if var_x2_key_next not in x:
            x[var_x2_key_next] = m.addVar(name=var_x2_key_next, vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=FUEL_CAPACITY)
        x[var_b_key] = m.addVar(name=var_b_key, vtype=gp.GRB.BINARY)

        m.addConstr(x[var_x2_key] >= x[var_x1_key])
        # m.addConstr(x["var_x2.." + str(sched[0])] >= x["var_x1.." + str(sched[n])])
        # m.addConstr(x["var_b.." + str(sched[n])]*properties.BIG_NUMBER >= x["var_x2.." + str(sched[n])] - x[
        # "var_x1.." + str(sched[n])]  )

        m.addConstr((x[var_b_key] == 1.0) >> (x[var_x2_key] - x[var_x1_key] >= 0.1))
        m.addConstr((x[var_b_key] == 0.0) >> (x[var_x2_key] - x[
            var_x1_key] == 0.0))  # due to very small numbers this forces some bunkering vars to be 1, despite a very
        # small thresh...

        m.addConstr(x[var_x2_key] - x[var_x1_next_key] == fuel_consume_n_next)
        # m.addConstr(x[var_x2_key] - x[var_x1_next_key] <= x[var_x2_key_next])
        # add in an extra constraint

    m.addConstr(x["var_x1.." + str(sched[0])] == 0)
    # m.addConstr(x["var_x1.." + str(sched[-1])] == 0)
    # m.addConstr(x["var_x2.." + str(sched[-1])] == 0)

    # m.addConstr(gp.quicksum([x2_coef_list[sched[n]]*x["var_x2.." + str(sched[n])] + x1_coef_list[sched[n]]*x[
    # "var_x1.." + str(sched[n])] + b_coef_list[sched[n]]*x["var_b.." + str(sched[n])]  for n in range(len(sched))])
    # >= 0) #should be a safety!
    m.setObjective(gp.quicksum([x2_coef_list[sched[n]] * x["var_x2.." + str(sched[n])] + x1_coef_list[sched[n]] * x[
        "var_x1.." + str(sched[n])] +
                                b_coef_list[sched[n]] * x["var_b.." + str(sched[n])] for n in range(len_)]),
                   gp.GRB.MINIMIZE)
    # + b_coef_list[sched[n]]*x["var_b.." + str(sched[n])]

    # print(properties.DIST_MAT)

    m.optimize()

    m.write("output/" + file_out_name)
    m.printAttr('x')

    data = json.loads(m.getJSONSolution())
    nz_sol = [(x['VarName'], x['X']) for x in data['Vars']]

    print(nz_sol)

    nz_sol_dic = dict(nz_sol)

    with open(filename_lp, 'wb') as handle:
        pkl.dump(nz_sol_dic, handle, protocol=pkl.HIGHEST_PROTOCOL)

    with open(filename_dm, 'wb') as handle:
        pkl.dump(DIST_MAT, handle, protocol=pkl.HIGHEST_PROTOCOL)

    print("wrote file to: " + str(filename_lp))
