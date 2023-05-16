import policies
from policies.policy_ss import *
from common import *
import os
import csv
from tqdm import tqdm

policy_library_name = 'policy_library_dpw'
def simulate(mariSim,
             schedule=ROUTE_SCHEDULE[1:],
             dist_mat=DIST_MAT,
             policy=policy_ss,
             print_bool=False,
             exp_arriv_time_rng=EXP_ARRIV_TIME_RNG,
             late_time_penalty=LATE_ARRIVAL_PENALTY,
             early_time_penalty=EARLY_ARRIVAL_PENALTY,
             simulation_saver_dict=None,
             simulation_number=None
             ):
    fuel_costs = []
    time_penalty_costs = []
    i = -1

    for n_to in schedule:#tqdm(schedule, desc='Schedule', leave=False, position=1):
        i += 1

        # assert len(schedule) == dist_mat.shape[0] -1, "Assertion Failed, Schedule length must equal to dist mat size"
        bunker_amt, speed = policy(mariSim)
        cum_f_before_cost = mariSim.cum_fuel_cost
        fuel_level_before = mariSim.fuel_level
        refuel_port = mariSim.current_n
        mariSim.refuelAtPort(bunker_amt, mariSim.current_n)
        departure_time = mariSim.time

        current_port = mariSim.current_n
        if n_to == ROUTE_SCHEDULE[-1]:
            n_to = 0
        if print_bool:
            print("TRAVERSAL #####")
            print("traversing from: " + str(mariSim.current_n) + " to: " + str(n_to))
            print("distance: " + str(dist_mat[mariSim.current_n, n_to]))
            print("bunkering amount: " + str(bunker_amt))
            print("Speed : " + str(speed))
            print("departure time from: " + str(mariSim.current_n) + " is at: " + str(departure_time))
        # bunkering_cost = mariSim.getFuelPrice(mariSim.current_n)*bunker_amt + np.clip(bunker_amt, 0,
        # 1)*mariSim.bunker_cost_callback(mariSim.current_n) ### MINUS the previous stored!!
        bunkering_cost = mariSim.cum_fuel_cost - cum_f_before_cost

        if print_bool:
            print("bunkering cost: " + str(bunkering_cost))
        fuel_costs.append(bunkering_cost)

        if print_bool:
            print("fuel level before: " + str(fuel_level_before))

        if simulation_saver_dict:
            simulation_saver_dict[
                f'{simulation_number}_FUEL LEVEL BEFORE: {mariSim.current_n}'] = mariSim.fuel_level
        if speed is not None:
            trav_bool = mariSim.traverse(n=ROUTE_SCHEDULE[i], n_to=n_to, speed=speed)
        else:
            trav_bool = mariSim.traverse(n=ROUTE_SCHEDULE[i], n_to=n_to)
        if mariSim.fuel_level == 14:
            a = 0
        if not trav_bool:
            break
        if print_bool:
            print("fuel level after: " + str(mariSim.fuel_level))
            print("arrival time at: " + str(mariSim.current_n) + " is at: " + str(departure_time + mariSim.travel_time))
            print("total time penalty: " + str(mariSim.time_penalty_cost))
        if simulation_saver_dict:
            simulation_saver_dict[f'{simulation_number}_ACTION _{current_port}_to_{n_to}'] = [
                f'Bunker Amt: {bunker_amt}', f'Bunkering Cost: {bunkering_cost}', f'Speed: {speed}',
                f'Fuel Level Before Travel: {fuel_level_before}', f'Fuel Level After Travel: {mariSim.fuel_level}']

        # time penalty cost:
        # if mariSim < 
        time_penalty_costs.append(mariSim.time_penalty_cost)
        new_row = [simulation_number, refuel_port, mariSim.getFuelPrice(refuel_port), fuel_level_before,
                   bunker_amt, speed, '', '']
        if not os.path.isfile(f'policies/{policy_library_name}.csv'):
            with open(f'policies/{policy_library_name}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['simulation_number', 'port', 'price', 'fuel_amount', 'refuel_amount', 'speed', 'fuel_cost', 'total_cost' ])
        with open(f'policies/{policy_library_name}.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(new_row)
    final_cost = np.sum(fuel_costs) + np.sum(time_penalty_costs)
    if simulation_saver_dict:
        simulation_saver_dict[f'{simulation_number}_FUEL COSTS'] = (mariSim.cum_fuel_cost)
        simulation_saver_dict[f'{simulation_number}_TIME PENALTY COSTS'] = np.sum(time_penalty_costs)
        simulation_saver_dict[f'{simulation_number}_FINAL COST'] = np.sum(fuel_costs) + np.sum(time_penalty_costs)
        simulation_saver_dict[
            f'{simulation_number}{simulation_number}{simulation_number}{simulation_number}_'] = 'FINISHED'

    with open(f'policies/{policy_library_name}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([simulation_number, '', '', '', '', '', mariSim.cum_fuel_cost, np.sum(time_penalty_costs)])
    if print_bool:
        print(f"##### FINAL COST {mariSim.seed} :" + str(mariSim.cum_fuel_cost))
        print("##### FINAL FUEL COST :" + str(mariSim.cum_fuel_cost))
        print("##### FINAL TIME PENALTY COST :" + str(np.sum(time_penalty_costs)))
        print("##### FINAL COST :" + str(final_cost))

    return final_cost, fuel_costs
