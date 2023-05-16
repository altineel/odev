import numpy as np
import gurobipy as gp
from common import *
from sim_environ.cost_functions import fuel_price_func, fuel_consume_const_func, fixed_bunkering_costs
import json
import pickle as pkl
from gurobipy import abs_


def solve_mip_fixv_stoch(dist_mat=DIST_MAT,
                         fuel_consume_rate=REGULAR_SPEED * DYN_VEL_FUEL_CONSUM_CONST,
                         fuel_price_list=EXPECTED_BUNKERING_COSTS,
                         fuel_capacity=FUEL_CAPACITY,
                         stoch_probs=STOCH_PROBS,
                         stoch_bunkering_costs=STOCH_BUNKERING_COSTS,
                         sched=ROUTE_SCHEDULE,
                         filename_sp=FILENAME_SP,
                         filename_dm=FILENAME_DM,
                         file_out_name="model_lp_mari_stoch.lp"):
    # GRB = gp.GRB
    assert np.shape(stoch_bunkering_costs)[0] == len(stoch_probs)
    assert np.abs(1 - np.sum(stoch_probs)) <= COMP_THRESH
    #print('STOCHASTIC BUNKERING COSTS', stoch_bunkering_costs)

    n = dist_mat.shape[0]
    #print(n)

    m = gp.Model("stoch_v2")
    x = {}
    x2_coef_dic = {}
    x1_coef_dic = {}
    b_coef_dic = {}

    len_ = len(sched) - 1
    for s in range(len(stoch_probs)):
        for n in range(len_):
            travel_dist_n_next = dist_mat[sched[n]][sched[n + 1]]

            p_n = fuel_price_func(sched[n], fuel_price_list=stoch_bunkering_costs[s])
            p_n_next = fuel_price_func(sched[n + 1], fuel_price_list=stoch_bunkering_costs[s])

            fuel_consume_n_next = fuel_consume_const_func(travel_dist_n_next)

            x2_coef = stoch_probs[s] * p_n
            x1_coef = stoch_probs[s] * (-p_n)  # + p_n_next
            b_coef = stoch_probs[s] * fixed_bunkering_costs(n)

            # x2_coef = p_n
            # x1_coef = -p_n # + p_n_next
            # b_coef = fixed_bunkering_costs(n)

            x2_coef_dic[repr([s, n])] = x2_coef
            x1_coef_dic[repr([s, n])] = x1_coef
            b_coef_dic[repr([s, n])] = b_coef

            var_x2_s_key = "var_x2_" + str(s) + ".." + str(sched[n])
            var_x1_s_key = "var_x1_" + str(s) + ".." + str(sched[n])
            var_x1_next_s_key = "var_x1_" + str(s) + ".." + str(sched[n] + 1)
            var_b_s_key = "var_b_" + str(s) + ".." + str(sched[n])

            # obj = coef 
            x[var_x2_s_key] = m.addVar(name=var_x2_s_key, vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=FUEL_CAPACITY)

            if var_x1_s_key not in x:
                x[var_x1_s_key] = m.addVar(name=var_x1_s_key, vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=FUEL_CAPACITY)
            if var_x1_next_s_key not in x:
                x[var_x1_next_s_key] = m.addVar(name=var_x1_next_s_key, vtype=gp.GRB.CONTINUOUS, lb=0.0,
                                                ub=FUEL_CAPACITY)

            x[var_b_s_key] = m.addVar(name=var_b_s_key, vtype=gp.GRB.BINARY)

            m.addConstr(x[var_x2_s_key] >= x[var_x1_s_key])

            # https://math.stackexchange.com/questions/2500415/how-to-write-if-else-statement-in-linear-programming
            # m.addConstr(x[var_x2_s_key] >= x[var_x1_s_key] + COMP_THRESH - BIG_NUMBER*(1-x[var_b_s_key]))
            # m.addConstr(x[var_x2_s_key] <= x[var_x1_s_key] + BIG_NUMBER*x[var_b_s_key])

            # m.addConstr(x[var_x2_s_key] - x[var_x1_s_key] >= COMP_THRESH - BIG_NUMBER*(1-x[var_b_s_key]))
            # m.addConstr(x[var_x2_s_key] - x[var_x1_s_key] <= COMP_THRESH + BIG_NUMBER*x[var_b_s_key]) #
            # BIG_NUMBER*x[var_b_s_key])

            m.addConstr((x[var_b_s_key] == 1.0) >> (x[var_x2_s_key] - x[var_x1_s_key] >= 0.1))
            m.addConstr((x[var_b_s_key] == 0.0) >> (x[var_x2_s_key] - x[
                var_x1_s_key] == 0.0))  # due to very small numbers this forces some bunkering vars to be 1,
            # despite a very small thresh...

            m.addConstr(x[var_x2_s_key] - x[var_x1_next_s_key] == fuel_consume_n_next)

        m.addConstr(x["var_x1_" + str(s) + ".." + str(sched[0])] == 0)
        # m.addConstr(x["var_x1_" + str(s) + ".." + str(sched[-1])] == 0)
        # m.addConstr(x["var_x2_" + str(s) + ".." + str(sched[-1])] == 0)

    # m.setObjective(gp.quicksum([x2_coef_dic[repr([s, n])]*x["var_x2_" + str(s) + ".." + str(sched[n])] +
    # x1_coef_dic[repr([s, n])]*x["var_x1_" + str(s) + ".." + str(sched[n])] for n in range(len(sched)) for s in
    # range(len(stoch_probs))]), gp.GRB.MINIMIZE)
    m.setObjective(gp.quicksum(
        [x2_coef_dic[repr([s, n])] * x["var_x2_" + str(s) + ".." + str(sched[n])] + x1_coef_dic[repr([s, n])]
         * x["var_x1_" + str(s) + ".." + str(sched[n])] + b_coef_dic[repr([s, n])] * x[
             "var_b_" + str(s) + ".." + str(sched[n])]
         for n in range(len_) for s in range(len(stoch_probs))]), gp.GRB.MINIMIZE)

    # print(DIST_MAT)

    m.optimize()

    m.write("output/" + file_out_name)
    #m.printAttr('x')

    obj = m.getObjective()
    vars = m.getAttr("X", m.getVars())

    with open("output/" + file_out_name, 'a') as the_file:
        the_file.write("sol obj: " + str(obj.getValue()) + "\n")
        the_file.write(str(vars) + "\n")

    data = json.loads(m.getJSONSolution())
    nz_sol = [(x['VarName'], x['X']) for x in data['Vars']]

    #print(nz_sol)

    nz_sol_dic = dict(nz_sol)

    with open(filename_sp, 'wb') as handle:
        pkl.dump(nz_sol_dic, handle, protocol=pkl.HIGHEST_PROTOCOL)

    with open(filename_dm, 'wb') as handle:
        pkl.dump(DIST_MAT, handle, protocol=pkl.HIGHEST_PROTOCOL)

    print("wrote file to: " + str(filename_sp))
