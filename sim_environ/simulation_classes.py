from sim_environ.cost_functions import *


# distance matrix
# node ordering n=0, to 1, assume all accessible

# for dynamic fuel prices, fuel_price_list
class MaritimeSim():
    def __init__(self, dist_mat,
                 fuel_price_func_callback=fuel_price_func,
                 fuel_consume_func_callback=fuel_consume_const_func,
                 bunker_cost_callback=fixed_bunkering_costs,
                 ini_fuel_level=0.0,
                 start_n=0,
                 cum_fuel_cost=0.0,
                 stoch_params=None,
                 fuel_capacity=FUEL_CAPACITY,
                 route_schedule=ROUTE_SCHEDULE[1:],
                 start_time=0.0,
                 time_penalty_cost=0.0,
                 exp_arrival_time_ranges=EXP_ARRIV_TIME_RNG,
                 late_arriv_penalty=LATE_ARRIVAL_PENALTY,
                 early_arriv_penalty=EARLY_ARRIVAL_PENALTY,
                 seed=None,
                 teu=7000
                 ):  # stoch_params in format [STOCH_PROBS, STOCH_BUNKERING_COSTS]

        self.dist_mat = dist_mat
        self.start_n = start_n
        self.current_n = self.start_n
        self.fuel_consume_rate = DYN_VEL_FUEL_CONSUM_CONST * REGULAR_SPEED
        self.fuel_level = ini_fuel_level
        self.cum_fuel_cost = cum_fuel_cost
        self.fuel_price_func_callback = fuel_price_func_callback
        self.fuel_consume_func_callback = fuel_consume_func_callback
        self.bunker_cost_callback = bunker_cost_callback
        self.n_size = self.dist_mat.shape[0]
        self.stoch_params = stoch_params
        self.n_iter = 0
        self.route_schedule = route_schedule
        self.fuel_capacity = fuel_capacity  # not implemented in stoch prog.
        self.time = start_time
        self.time_penalty_cost = time_penalty_cost
        self.exp_arrival_time_ranges = exp_arrival_time_ranges
        self.late_arriv_penalty = late_arriv_penalty
        self.early_arriv_penalty = early_arriv_penalty
        self.abs_n_counter = 0
        self.seed = seed
        self.teu = teu
        self.k1, self.k2 = k_coefficients(self.teu)

        assert self.dist_mat.shape[1] == self.dist_mat.shape[
            0], "Assertion Failed: distance matrix must be square n == n."
        assert self.stoch_params == None or len(self.stoch_params) == 3, "Assertion Failed: invalid stoch_params!"

        self.cur_fuel_prices = None
        if self.stoch_params is not None and PRICE_DISTRIBUTION == 'multinomial':
            if seed:
                np.random.seed(seed)
            poss_scenarios = list(range(len(self.stoch_params[1])))
            probabilities = self.stoch_params[0]
            sc = np.random.choice(poss_scenarios, 1, p=probabilities)[0]
            self.multiplied_scenarios = self.stoch_params[2][sc]
            self.sc = sc
            fuel_prices_vals = self.stoch_params[1]
            self.cur_fuel_prices = fuel_prices_vals[sc]
            print('-------- FUEL PRICES ---------')
            print(self.cur_fuel_prices)
            print('------------------------------')

        elif self.stoch_params is not None and PRICE_DISTRIBUTION == 'discrete_normal':
            raise Exception('discrete normal is not defined yet ! ')
            # self.cur_fuel_prices = [general_fuel_price_function(n=i, means=EXPECTED_BUNKERING_COSTS,
            #                                                     stds=PRICE_STD) for i in range(4)]

    def getFuelPrice(self, n):
        if self.stoch_params is not None:
            return self.fuel_price_func_callback(n, fuel_price_list=self.cur_fuel_prices)
        else:
            return self.fuel_price_func_callback(n)

    def distFunc(self, n_from, n_to):
        assert n_from < self.dist_mat.shape[0] and n_to < self.dist_mat.shape[
            1], "Assertion Failed: quert n_from or n_to greater than dist matrix."
        dist_val = self.dist_mat[n_from, n_to]
        return dist_val

    def refuelAtPort(self, fill_amount, n):
        self.fuel_level += fill_amount
        fix_bunker_cost = self.bunker_cost_callback(n) if fill_amount > 0.0 else 0.0
        self.cum_fuel_cost += self.getFuelPrice(n) * fill_amount
        self.cum_fuel_cost += fix_bunker_cost
        return self.fuel_level

    def traverse(self, n, n_to, speed=REGULAR_SPEED, dyn_fuel_param=DYN_VEL_FUEL_CONSUM_CONST,
                 speed_power_coef=SPEED_POWER_COEF):
        assert n_to < self.n_size, "Assertion Failed: n_from or n_to greater than dist matrix size."
        travel_dist = self.distFunc(self.current_n, n_to)
        travel_time = travel_time_function(travel_dist, speed)
        #fuel_cons_rate =

        # fuel_loss = self.fuel_consume_func_callback(travel_dist, fuel_consume_rate = dyn_fuel_cons_rate)
        fuel_loss = fuel_consumption_function(travel_dist, speed, self.k1, self.k2)
        new_fuel_level = self.fuel_level - fuel_loss

        # check legality of traverse
        # if np.abs(new_fuel_level) > COMP_THRESH and new_fuel_level < 0:
        #     print("traversal invalid, not enough fuel")
        #     return False

        if new_fuel_level < 0 - SMALL_NUMBER:
            print("traversal invalid, not enough fuel")
            return False

        self.fuel_level = new_fuel_level
        self.current_n = n_to
        self.n_iter += 1
        self.travel_time = 0
        if USE_SPEED:
            self.travel_time += travel_time
            if self.time < self.exp_arrival_time_ranges[self.abs_n_counter][0]:
                self.time_penalty_cost += self.early_arriv_penalty
            elif self.time > self.exp_arrival_time_ranges[self.abs_n_counter][1]:
                self.time_penalty_cost += self.late_arriv_penalty

        self.time += self.travel_time + get_port_time(n)
        self.abs_n_counter += 1
        # if self.stoch_params is not None:
        #     self.sc = np.random.choice(range(len(self.stoch_params[1])), 1, p=self.stoch_params[0])[0]
        #     self.cur_fuel_prices = self.stoch_params[1][self.sc]

        return True

    def getPossPorts(self, fuel_add=0):
        travel_dists = [self.distFunc(self.current_n, x) for x in range(self.n_size)]
        return [self.fuel_level + fuel_add - self.fuel_consume_func_callback(travel_dist) >= 0 for travel_dist in
                travel_dists]
