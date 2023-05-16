from common.properties import BIG_S, CUM_TRAVEL_TIME, DYN_VEL_FUEL_CONSUM_CONST, DIST_MAT, EXP_ARRIV_TIME_RNG, \
    EXPECTED_BUNKERING_COSTS, EXP_TRAVEL_TIME, REGULAR_SPEED, REL_TIME_DIFFS, ROUTE_SCHEDULE, SMALL_S, \
    STOCH_BUNKERING_COSTS, STOCH_PROBS, FILENAME_SP, FILENAME_DM, FILENAME_LP, k_coefficients, get_terminal_state, \
    dist_matrix, save_config,  PRICE_SCENARIOS, PRICE_PERCENTAGES, FUEL_CAPACITY, FIXED_BUNKERING_COSTS
from datetime import datetime

BIG_NUMBER = 9e99
BIG_S = BIG_S
COMP_THRESH = 1e-3
DECREASING_ITER_NUMBERS = True
DISCOUNT_FACTOR = 1
DIST_MAT = DIST_MAT
DPW_ALPHA= 0.9
DPW_EXPLORATION = 0.9

EARLY_ARRIVAL_PENALTY = 0
EARLY_STOP = False
EPSILON = 0.001
FUEL_SAFETY_FACTOR = 1
HEURISTIC = False
INITIAL_SPEED = 1
KAPPA = 1.2
KAPPA2 = 1.1
LATE_ARRIVAL_PENALTY = 50.0
LAST_ITERATIONS_NUMBER = 100
MCTS_DISCRETE_CARDINAL = 15
MAX_ITERATION = 500
MAX_SPEED = 10
MIN_ITERATION = 500
MIN_SPEED = 1
MIN_FUEL_ALLOWANCE = 0
NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
POLICY_LIBRARY = False
PRICE_DISTRIBUTION = 'multinomial'
PRICE_STD = [50, 50, 50, 50, 50]
SIMULATION_DEPTH_LIMIT = 15
SMALL_NUMBER = 1e-5
SMALL_S = SMALL_S
USE_SPEED = False
SPEED_POWER_COEF = 1.0
TEU = 7000
MIN_SET_SIZE_REFUEL = 5#round(FUEL_CAPACITY/100)
MIN_SET_SIZE_SPEED = 5
MIN_SET_SIZE = MIN_SET_SIZE_SPEED * MIN_SET_SIZE_REFUEL
USE_TEU = False
verbose = False
#DECREASING_EXPLORATION_FACTOR = 0.95

FUEL_COST_METHOD = 'SIMPLE' # NONLINEAR
#ACTION_METHOD = 'PW'

import os
import pandas as pd
import json
def parser(now, path,simulation_name=None):
    # path = locals()['__file__']
    # path = path[:path.find('common')]
    if simulation_name:
        path = path + f'/runs/{now[:10]}/{simulation_name}'
    else:
        path = path + f'/runs/{now[:10]}/{now}'
    files = os.listdir(path)
    configs_jsons = []
    simulations_jsons = []
    dataframe = pd.DataFrame()
    numbers = []
    for f in files:
        if f.startswith('config'):
            number = f.split('_')[1]
            numbers.append(number.split('.')[0])
            with open(os.path.join(path, f)) as json_file:
                configs_jsons.append(json.load(json_file))
            with open(os.path.join(path, f'simulations_{number}')) as json_file:
                simulations_jsons.append(json.load(json_file))
    total_simulations = len(simulations_jsons)
    different_configs = []
    for keys in configs_jsons[0]:
        for configs in configs_jsons:
            try:
                if configs_jsons[0][keys] != configs[keys]:
                    different_configs.append(keys)
            except:
                continue
    for t in range(total_simulations):
        local_dataframe = pd.DataFrame({'Simulation number': numbers[t]}, index=[0])
        for dif in different_configs:
            if dif =='run_time':
                local_dataframe[dif] = int(configs_jsons[t][dif])
            else:
                local_dataframe[dif] = configs_jsons[t][dif]
        total_runs = 0
        total_cost = 0
        total_time_penalty_cost = 0
        total_fuel_cost = 0
        for i in simulations_jsons[t]:
            if i.startswith('----- Simulation'):
                total_cost += simulations_jsons[t][f'{int(numbers[t])*100+total_runs}_FINAL COST']
                total_time_penalty_cost += simulations_jsons[t][f'{int(numbers[t])*100+total_runs}_TIME PENALTY COSTS']
                total_fuel_cost += simulations_jsons[t][f'{int(numbers[t])*100+total_runs}_FUEL COSTS']
                total_runs += 1
        local_dataframe['Average_Cost'] = round(total_cost / total_runs)
        local_dataframe['Total_Time_Penalty_Cost'] = round(total_time_penalty_cost / total_runs)
        local_dataframe['Total_Fuel_Cost'] = round(total_fuel_cost / total_runs)
        dataframe = dataframe.append(local_dataframe, ignore_index=True)
    print('SIMULATION RESULT: ',  total_cost / total_runs)
    dataframe.to_csv(path + '/results.csv', index=False)

from copy import copy
locals = copy(locals())
_path = locals['__file__']
_index = _path.find('common')
PATH = _path[:_index]
def save_configs(name,simulation_name=None, **kwargs):
    variables = {}
    variables['BIG_S'] = BIG_S
    variables['DECREASING_ITER_NUMBERS'] = DECREASING_ITER_NUMBERS
    variables['DISCOUNT_FACTOR'] = DISCOUNT_FACTOR
    variables['DIST_MAT'] = DIST_MAT
    variables['DPW_ALPHA'] = DPW_ALPHA
    variables['DPW_EXPLORATION'] = DPW_EXPLORATION
    #variables['EARLY_STOP'] = EARLY_STOP
    variables['EPSILON'] = EPSILON
    variables['FIXED_BUNKERING_COSTS'] = FIXED_BUNKERING_COSTS
    variables['FUEL_CAPACITY'] = FUEL_CAPACITY
    variables['LAST_ITERATIONS_NUMBER'] = LAST_ITERATIONS_NUMBER
    variables['MAX_ITERATION'] = MAX_ITERATION
    variables['MIN_ITERATION'] = MIN_ITERATION
    variables['POLICY_LIBRARY'] = POLICY_LIBRARY
    variables['PRICE_DISTRIBUTION'] = PRICE_DISTRIBUTION
    variables['SIMULATION_DEPTH_LIMIT'] = SIMULATION_DEPTH_LIMIT
    variables['USE_SPEED'] = USE_SPEED
    variables['SPEED_POWER_COEF'] = SPEED_POWER_COEF
    variables['TEU'] = TEU
    variables['MIN_SET_SIZE_REFUEL'] = MIN_SET_SIZE_REFUEL
    variables['MIN_SET_SIZE_SPEED'] = MIN_SET_SIZE_SPEED
    variables['USE_TEU'] = USE_TEU
    #variables['DECREASING_EXPLORATION_FACTOR'] = DECREASING_EXPLORATION_FACTOR
    for m in kwargs:
        variables[m] = kwargs[m]
    if simulation_name:
        save_config(variables, name=f'config_{name}', path=PATH, now=NOW, simulation_name=simulation_name)
    else:
        save_config(variables, name=f'config_{name}', path=PATH, now=NOW)
# save_config(locals(), 0, NOW)
if USE_SPEED:
    assert len(EXP_ARRIV_TIME_RNG) == len(
        ROUTE_SCHEDULE) - 1, "Assert: ensure there are enough time constraints per port in the schedule."
