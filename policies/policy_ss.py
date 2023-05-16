from sim_environ.cost_functions import *
from common import *
route_schedule = ROUTE_SCHEDULE
dist_mat = DIST_MAT

# max_S = np.max(dist_mat) * FUEL_CONSUM_CONST*1.1
max_S = BIG_S
small_s = SMALL_S
s_s_tuple = [small_s, max_S]

# state rep is tuple (port, next_port_distance)
def policy_ss(mariSim,
    s_s_tuple = s_s_tuple,
    fuel_cap = FUEL_CAPACITY):
    
    assert s_s_tuple[0] < s_s_tuple[1], "Assertion failed: small_s must be < big_S"
    fuel_level = mariSim.fuel_level

    if fuel_level < s_s_tuple[0]:
        bunker_amt = np.min([s_s_tuple[1], fuel_cap]) - fuel_level
    else:
        bunker_amt = 0.0
    
    speed = 1
    return bunker_amt, speed

#run mcts in the policy again. each times when you run simulation, you need to rerun the mcts . and get the best uct action

