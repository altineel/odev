import numpy as np
from itertools import product, repeat
from operator import mul
import datetime
import time
import os
import json

import pandas as pd

FILENAME_LP = "output/lp_sol.pkl"
FILENAME_SP = "output/sp_sol.pkl"
FILENAME_DM = "output/dist_mat.pkl"
# Speed variables
REGULAR_SPEED = 1.0
DYN_VEL_FUEL_CONSUM_CONST = 1.0  # 0.04
FUEL_CAPACITY = 200
FIXED_BUNKERING_COSTS =    [7000, 5000, 1000, 3500, 2000, 1500, 1000, 7000, 4500, 8000, 4900, 5800, 6100, 7500]
EXPECTED_BUNKERING_COSTS = [5000, 6000, 2000, 5500, 8000, 4500, 3500, 3000, 2100, 2900, 3400, 4700, 3800, 2600]
COMP_THRESH = 15
ROUTE_SCHEDULE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  0]  # first value does not matter.

# Non-Euclidean / Symmetric distance matrix
DIST_MAT = np.array([[0, 30, 7, 15, 100, 100, 100, 100, 100, 100],
                     [30, 0, 25, 8, 100, 100, 100, 100, 100, 100],
                     [7, 25, 0, 30, 100, 100, 100, 100, 100, 100],
                     [15, 8, 30, 0, 25, 100, 100,  100, 100, 100],
                     [15, 8, 30, 100, 0, 33, 100,  100, 100, 100],
                     [15, 8, 30, 100, 100, 0, 45,   44, 100, 100],
                     [45, 8, 30, 100, 100, 100, 0,  22, 100, 100],
                     [15, 8, 30, 100, 100, 100, 100, 0, 65,  100],
                     [15, 8, 30, 100, 100, 100, 100, 100, 0,  55],
                     [25, 8, 30, 100, 100, 100, 100, 100, 100,  0],
                     ])
BIG_S = np.max(DIST_MAT) * DYN_VEL_FUEL_CONSUM_CONST * 6
SMALL_S = BIG_S * 0.95
EXP_TRAVEL_TIME = np.array(
    [30.0, 25.0, 30.0, 15.0, 30.0, 25.0, 30.0, 15.0, 30.0, 25.0  # , 30.0, 15.0, 30.0
     ]) / REGULAR_SPEED,


# s0 is fixed
def generate_prices_n_scenarios(p0=EXPECTED_BUNKERING_COSTS, s0=[1], sn=[0.75, 1, 1.25]):
    l = len(p0)
    scenarios = list(product(s0, *repeat(sn, l - 1)))
    prices = []
    scc = []
    for s in scenarios:
        c = [1]
        for delta in s:
            c.append(c[-1] * delta)
        c.pop(0)
        p = list(map(mul, p0, c))
        sc = list(map(mul, [1 for i in p0], c))
        prices.append(p)
        scc.append(sc)

    return prices, scc


PRICE_PERCENTAGES = [0.75, 1, 1.25]
#PRICE_PERCENTAGES =[0.75, 1, 1.25]
# stochastic parameters, symmetric expected costs
STOCH_BUNKERING_COSTS, PRICE_SCENARIOS = generate_prices_n_scenarios(p0=EXPECTED_BUNKERING_COSTS,
                                                                     s0=[1],
                                                                     sn=PRICE_PERCENTAGES)

# def generate_probabilties(costs=STOCH_BUNKERING_COSTS, scenarios=PRICE_SCENARIOS):
#     """
#     :param costs:
#     :param scenarios:
#     :return:
#
#     """
#     len_cost = len(costs)
#     all_percentages = [scenarios[i][1] for i in range(len_cost)]
#     all_percentages = set(all_percentages)
#     assert len(all_percentages) in (3,5), "There could be only 3 or 5 different percentages"
#     transition_csv=pd.read_csv(f"common/transition_{len(all_percentages)}sc.csv")
#     number_of_ports = len(scenarios[0])
#     first_scenarios = [scenarios[i][0] for i in range(len_cost)] # to check if all scenarios are the same,
#     then ignore it.
#     start_port = 1 if set(first_scenarios) == 1 else 0 #it is for our case where we always start with the same
#     scenario
#     for scenario in scenarios:
#         perc = 1
#         for port in (range(start_port, number_of_ports)):
#             perc *= transition_csv[scenario[port]][scenario[port+1]]
# Equal occurence probabilities
STOCH_PROBS = [1 / len(STOCH_BUNKERING_COSTS)] * len(STOCH_BUNKERING_COSTS)

# Time frame variables
CUM_TRAVEL_TIME = np.cumsum(EXP_TRAVEL_TIME)
REL_TIME_DIFFS = np.array(
    [[-3, 2], [-2, 1], [-3, 3], [-2, 4], [-3, 2], [-2, 1], [-3, 3], [-2, 4], [-3, 2], [-2, 1],
     # [-3, 3], [-3, 2], [-2, 1]
     ])
EXP_ARRIV_TIME_RNG = [[x[0] + x[1][0], x[0] + x[1][1]] for x in zip(CUM_TRAVEL_TIME, REL_TIME_DIFFS)]

EARLY_ARRIVAL_PENALTY = 0.0  # 500.0

# assert len(EXP_ARRIV_TIME_RNG) == len(
#     ROUTE_SCHEDULE) - 1, "Assert: ensure there are enough time constraints per port in the schedule."


# speed quadratic relation
# know the branching of the MCTS tree, and ensure iters and cardinality are propotional


def k_coefficients(teu):
    if teu <= 1000:
        k1 = 0.004476
        k2 = 6.17
    elif teu <= 2000:
        k1 = 0.004595
        k2 = 16.42
    elif teu <= 3000:
        k1 = 0.004501
        k2 = 29.28
    elif teu <= 4000:
        k1 = 0.006754
        k2 = 32.23
    elif teu <= 5000:
        k1 = 0.006732
        k2 = 55.84
    elif teu <= 6000:
        k1 = 0.007297
        k2 = 71.4
    else:
        k1 = 0.006705
        k2 = 87.71
    return k1, k2


def get_terminal_state(departure_time, port):  # change where it is used to use this function
    if departure_time not in[0,1]:
        return port == ROUTE_SCHEDULE[-1]
    else:
        return False


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if callable(obj):
            return obj.__name__
        return super(NpEncoder, self).default(obj)


def save_config(args, name, path, now, simulation_name=None):
    from copy import copy

    keys = copy(list(args.keys()))
    for i in keys:
        if i.startswith('__'):
            del args[i]
        elif callable(args[i]):
            del args[i]
        elif isinstance(args[i], type(np)):
            del args[i]
    if simulation_name:
        path = f"{path}/runs/{now[:10]}/{simulation_name}/"
    else:
        path = f"{path}/runs/{now[:10]}/{now}/"
    if not os.path.exists(path):
        os.makedirs(path)
    pathrunconfig = path + name + '.json'
    with open(pathrunconfig, 'w') as fp:
        json.dump(args, fp, indent=4, cls=NpEncoder),


def dist_matrix(start, end):
    if start == (len(DIST_MAT)-1) and end == (len(DIST_MAT)):
        return DIST_MAT[start, 0]
    return DIST_MAT[start, end]
