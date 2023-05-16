from sim_environ.cost_functions import *
from common.properties import FILENAME_SP
import pickle as pkl


# state rep is tuple (port, next_port_distance)
def policy_sp(mariSim, sp_filename=FILENAME_SP, stoch_probs=STOCH_PROBS, print_warn_bool=True):
    with open(sp_filename, 'rb') as f:
        sp_sol_dic = pkl.load(f)

    fuel_level = mariSim.fuel_level
    n_port = mariSim.current_n
    sc = mariSim.sc

    # rec_fuel_lvl = 0
    # for sc in range(len(stoch_probs)):
    key_dic_x2 = "var_x2_" + str(sc) + ".." + str(n_port)
    scenario_amt = sp_sol_dic[key_dic_x2] if key_dic_x2 in sp_sol_dic else 0.0
    rec_fuel_lvl = scenario_amt  # * stoch_probs[sc]

    bunker_amt = np.clip(rec_fuel_lvl - fuel_level, 0, np.Inf)

    key_dic_vel = "var_vel_" + str(sc) + ".." + str(n_port)
    rec_speed = sp_sol_dic[key_dic_vel] if key_dic_vel in sp_sol_dic else 1.0

    # trip_fuel = bunker_amt + fuel_level

    # dist = mariSim.distFunc(mariSim.current_n, ROUTE_SCHEDULE[mariSim.current_n + 1])
    # f_consump = mariSim.fuel_consume_func_callback(dist)
    # bunker_amt = rec_fuel_lvl - fuel_level

    if print_warn_bool:
        if rec_fuel_lvl - fuel_level < 0:
            print("WARNING, Bunker amt negative: " + str(bunker_amt) + "/" + str(rec_fuel_lvl - fuel_level))
            # return False

        if key_dic_x2 not in sp_sol_dic:
            print("WARNING, " + str(key_dic_x2) + " not in SP solution!")

        #if key_dic_vel not in sp_sol_dic:
         #   print("WARNING, " + str(key_dic_vel) + " not in SP solution!")

    return bunker_amt, rec_speed