import numpy as np
from common import *
from math import floor


# designed to support dynamic/stochastic fuel prices
def fuel_price_func(n, fuel_price_list=EXPECTED_BUNKERING_COSTS):
    fuel_price = fuel_price_list[n]
    return fuel_price


def general_fuel_price_function(n, seed=None, stds=None, means=None, previous_perc=1):
    """

    :param n:
    :param seed: To make sure when we iterate over the nodes, we get the same fuel prices
    :param stds:
    :param means:
    :param previous_perc: If previous perc =1 then we are in the medium market and we use the medium prices otherwise
    we adjust the prices according to the previous perc
    :return:
    """
    if seed:
        np.random.seed(floor(seed))
    if PRICE_DISTRIBUTION == 'multinomial':
        if n == 0:
            return fuel_price_func(n), 1
        new_perc = np.random.choice([0.75, 1, 1.25]) * previous_perc
        return fuel_price_func(n) * new_perc, new_perc
    elif PRICE_DISTRIBUTION == 'discrete_normal':
        if stds is None:
            stds = PRICE_STD
        if means is None:
            means = EXPECTED_BUNKERING_COSTS
        mean = means[n]
        prices_std = stds[n]
        return np.random.choice((np.random.normal(mean, prices_std, 1000)).astype(int))


def fuel_consume_const_func(travel_distance, speed=1, k1=1000, k2=1000,
                            fuel_consume_rate=DYN_VEL_FUEL_CONSUM_CONST * REGULAR_SPEED):
    # fuel_loss = fuel_consume_rate * travel_distance
    assert FUEL_COST_METHOD in ['NONLINEAR', 'SIMPLE']
    if FUEL_COST_METHOD =='NONLINEAR':
        fuel_loss = (k1 * pow(speed, 3) + k2) * travel_distance / 24
    elif FUEL_COST_METHOD == 'SIMPLE':
        fuel_loss = travel_distance * DYN_VEL_FUEL_CONSUM_CONST
    return fuel_loss


def fixed_bunkering_costs(n, fixed_bunk_costs=FIXED_BUNKERING_COSTS):
    fixed_bunker_cost = fixed_bunk_costs[n]
    return fixed_bunker_cost


def fuel_consumption_function(distance, speed, k1, k2):  # tons per day
    assert FUEL_COST_METHOD in ['NONLINEAR', 'SIMPLE']
    if FUEL_COST_METHOD =='NONLINEAR':
        consumption_per_day = k1 * pow(speed, 3) + k2
        total_consumption = consumption_per_day * (distance / speed) / 24
    elif FUEL_COST_METHOD == 'SIMPLE':
        total_consumption = distance * DYN_VEL_FUEL_CONSUM_CONST
    return total_consumption


def travel_time_function(distance, speed):
    randomness = np.random.normal(1, 0.05)
    randomness = 1
    # WEATHER CONDITIONS CAN BE ADDED HERE
    return round((randomness * distance) / speed)


def get_port_time(port):
    # SET RANDOMIZED PORT TIME DEPENDING ON PORT
    return 1
