from sim_environ.cost_functions import *
from common.properties import FILENAME_LP
import pickle as pkl


# state rep is tuple (port, next_port_distance)
def policy_lp(mariSim, lp_filename = FILENAME_LP, print_warn_bool = True):

    with open(lp_filename, 'rb') as f:
        lp_sol_dic = pkl.load(f)
    
    fuel_level = mariSim.fuel_level
    n_port = mariSim.current_n
    key_dic = "var_x2.." + str(n_port)

    rec_fuel_lvl = lp_sol_dic[key_dic] if key_dic in lp_sol_dic else 0.0
    bunker_amt = np.clip(rec_fuel_lvl - fuel_level, 0, np.Inf)
    # bunker_amt = rec_fuel_lvl - fuel_level

    if print_warn_bool:
        if rec_fuel_lvl - fuel_level < 0:
            print("WARNING, Bunker amt negative: " + str(bunker_amt) + "/" + str(rec_fuel_lvl - fuel_level))
            # return False

        if key_dic not in lp_sol_dic:
            print("WARNING, " + str(key_dic) + " not in LP solution!")
    
    speed = None
    return bunker_amt, speed
