import os

from sim_environ.simulation_classes import *
from sim_environ.cost_functions import *
from sim_environ.simulator import *
from common import *
import itertools
import time
import policies
from solver.mip_stoch_solve import *
from solver.mip_deter_solve import *
from tqdm import tqdm
import random
from policies.policy_lp import *
from policies.policy_sp import *

dist_mat = DIST_MAT

# with open(FILENAME_DM, 'rb') as f:
#     dist_mat = pkl.load(f)

schedule = ROUTE_SCHEDULE[1:]

# solve_mip_stoch()
# solve_mip_deter()

costs_sp = []
fuel_costs_sp = []

costs_lp = []
fuel_costs_lp = []


def stochastic_solutions():
    for i in range(999):
        mariSimSP = MaritimeSim(dist_mat=dist_mat,
                                fuel_price_func_callback=fuel_price_func,
                                fuel_consume_func_callback=fuel_consume_const_func,
                                stoch_params=[STOCH_PROBS, STOCH_BUNKERING_COSTS])

        final_cost_sp, fuel_costs_trip_sp, trav_bool_sp = simulate(mariSimSP, schedule=ROUTE_SCHEDULE[1:],
                                                                   dist_mat=DIST_MAT, policy=policy_sp)
        fuel_costs_sp.append(np.sum(fuel_costs_trip_sp))
        costs_sp.append(final_cost_sp)

        mariSimLP = MaritimeSim(dist_mat=dist_mat,
                                fuel_price_func_callback=fuel_price_func,
                                fuel_consume_func_callback=fuel_consume_const_func,
                                stoch_params=[STOCH_PROBS, STOCH_BUNKERING_COSTS])

        final_cost_lp, trip_fuel_costs_lp, trav_bool_lp = simulate(mariSimLP, schedule=ROUTE_SCHEDULE[1:],
                                                                   dist_mat=DIST_MAT, policy=policy_lp)
        fuel_costs_lp.append(np.sum(trip_fuel_costs_lp))
        costs_lp.append(final_cost_lp)

    mean_costs_sp = np.mean(costs_sp)
    mean_fuel_costs_sp = np.mean(fuel_costs_sp)

    print(f'Stochastic solution: {mean_costs_sp}')
    print(f'Stochastic time cost solution: {mean_costs_sp - mean_fuel_costs_sp}')
    print(f'Stochastic fuel solution: {mean_fuel_costs_sp}')

    mean_costs_lp = np.mean(costs_lp)
    mean_fuel_costs_lp = np.mean(fuel_costs_lp)

    print(f'Deterministic solution: {mean_costs_lp}')
    print(f'Deterministic time cost solution: {mean_costs_lp - mean_fuel_costs_lp}')
    print(f'Deterministic fuel solution: {mean_fuel_costs_lp}')


def statefulsolver(name, simulation_name, **kwargs):
    start_time = time.time()
    simulations = {}
    nbsimulations = 25
    for i in tqdm(range(nbsimulations), desc='Run', position=0):
        mariSimLP = MaritimeSim(dist_mat=DIST_MAT,
                                fuel_price_func_callback=fuel_price_func,
                                fuel_consume_func_callback=fuel_consume_const_func,
                                stoch_params=[STOCH_PROBS, STOCH_BUNKERING_COSTS, PRICE_SCENARIOS],
                                seed=i + 1, teu=TEU
                                )
        simulations[f'{i}{i}{i}{i}'] = 'STARTED'
        simulations[f'----- Simulation {i} -----'] = name * 100 + i
        simulations[f'{name * 100 + i}_FUEL PRICES '] = mariSimLP.cur_fuel_prices

        final_cost_lp, _ = simulate(mariSimLP, print_bool=False, policy=policies.policy_mcts_stateful,
                                    simulation_saver_dict=simulations, simulation_number=name * 100 + i)
        print(f'Cost Solution Stateful: {final_cost_lp}')
    run_time = (time.time() - start_time)
    print("--- %s seconds ---" % run_time)
    kwargs['number_of_simulations'] = nbsimulations
    save_configs(run_time=run_time, name=name, simulation_name=simulation_name, **kwargs)
    save_config(simulations, name=f'simulations_{name}', path=PATH, now=NOW, simulation_name=simulation_name)


def linear_solver():
    # mariSimSP = MaritimeSim(dist_mat=dist_mat,
    #                         fuel_price_func_callback=fuel_price_func,
    #                         fuel_consume_func_callback=fuel_consume_const_func,
    #                         stoch_params=[STOCH_PROBS, STOCH_BUNKERING_COSTS])
    #
    # mariSimLP = MaritimeSim(dist_mat=dist_mat,
    #                         fuel_price_func_callback=fuel_price_func,
    #                         fuel_consume_func_callback=fuel_consume_const_func,
    #                         stoch_params=[STOCH_PROBS, STOCH_BUNKERING_COSTS])

    # final_cost_sp, _ = simulate(mariSimSP, schedule=ROUTE_SCHEDULE[1:], dist_mat=DIST_MAT,
    #                             policy=policy_sp, print_bool=True)
    mariSimLP = MaritimeSim(dist_mat=DIST_MAT,
                            fuel_price_func_callback=fuel_price_func,
                            fuel_consume_func_callback=fuel_consume_const_func,
                            stoch_params=[STOCH_PROBS, STOCH_BUNKERING_COSTS]
                            )
    simulate(mariSimLP, schedule=ROUTE_SCHEDULE[1:], dist_mat=DIST_MAT,
             policy=policies.policy_mcts_stateful, print_bool=True)


if __name__ == '__main__':
    random.seed(time.time())
    run_id = random.randint(0, 100000)
    i = run_id * 100
    simulation_name = '10portAlgComparison'
    for sim_depth_limit in [10]:
        for decreasing_iter in [False]:
            for early_stop in [False]:
                for exp_const in [500000, 800000]:  # , 12000, 15000, 20000]:
                    for max_iter in [300,500]:
                        for expl_const_decay in [0.9998]:  # , 0.9993, 0.9991]:
                            for alg in ['DPWSolver', 'ProgressiveWideningSolver']:
                                for opt_act in ['MIN_REWARD']:
                                    if early_stop == True:
                                        os.environ['EARLY_STOP'] = str(early_stop)
                                    if max_iter ==6000:
                                        decreasing_iter = True
                                    print(i - run_id * 100)
                                    os.environ['ALGORITHM'] = alg
                                    os.environ['SIMULATION_DEPTH_LIMIT'] = str(sim_depth_limit)
                                    os.environ['EXPLORATION_CONSTANT'] = str(exp_const)
                                    os.environ['MAX_ITERATION'] = str(max_iter)
                                    os.environ['EARLY_STOP'] = str(early_stop)
                                    os.environ['DECREASING_ITER'] = str(decreasing_iter)
                                    os.environ['OPTIMAL_ACTION_POLICY'] = str(opt_act)
                                    os.environ['EXP_CONST_DECAY'] = str(expl_const_decay)
                                    if alg == 'NAIVE':
                                        i += 1
                                        os.environ['FORCE_0_FUEL'] = str(False)
                                        os.environ['DPW_EXPLORATION'] = str(0)
                                        os.environ['DPW_ALPHA'] = str(0)
                                        statefulsolver(i, simulation_name=simulation_name,
                                                       **os.environ)
                                    else:
                                        for dpw_enum, dpw_alpha_refuel in enumerate(
                                                [0.4]):  # , 0.8, 0.9, 0.95]:
                                            for dpw_enum2, dpw_exploration_refuel in enumerate(
                                                    [4]):  # , 0.8, 0.9, 0.95]:
                                                for force in [False, True]:
                                                    i += 1
                                                    os.environ['FORCE_0_FUEL'] = str(force)
                                                    os.environ['DPW_EXPLORATION'] = str(
                                                        dpw_exploration_refuel)
                                                    os.environ['DPW_ALPHA'] = str(dpw_alpha_refuel)
                                                    statefulsolver(i, simulation_name=simulation_name, **os.environ)
    parser(now=NOW, path=PATH, simulation_name=simulation_name)
