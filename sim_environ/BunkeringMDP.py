from __future__ import annotations
import time
from sim_environ.cost_functions import *
from common import *
import math
from mcts4py_new.MDP import *
import os


class BunkeringAction():
    def __init__(self, speed, refuel_amount, next_port):
        self.speed, self.refuel_amount, self.next_port = speed, refuel_amount, next_port

    def __str__(self) -> str:
        return f"Speed: {self.speed}, Ref: {self.refuel_amount}, NextPort: {self.next_port}, 'Speed : {self.speed}"

    def __eq__(self, __o):
        if isinstance(__o, BunkeringAction):
            return __o.speed == self.speed and __o.refuel_amount == self.refuel_amount and __o.next_port == \
                   self.next_port
        return False

    def __hash__(self) -> int:
        return hash(self.__str__())

    @property
    def name(self):
        return f"Rf_{self.refuel_amount}_S{self.speed}"


class BunkeringState():
    def __init__(self, port: int, fuel_amount, arrival_time, speed=INITIAL_SPEED, price_perc=1, price=None):
        self.port = port
        self.fuel_amount = fuel_amount
        self.speed = speed
        self.price = fuel_price_func(port) * price_perc if price is None else price
        self.price_perc = price_perc
        self.fixed_bcost = fixed_bunkering_costs(port)
        self.arrival_time = arrival_time
        self.departure_time = self.arrival_time + get_port_time(self.port)
        self.is_terminal = get_terminal_state(self.departure_time, self.port)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, BunkeringState):
            return __o.port == self.port and __o.fuel_amount == self.fuel_amount and __o.speed == self.speed and \
                   __o.arrival_time == self.arrival_time and __o.price == self.price
        return False

    def __str__(self) -> str:
        return f'Port: {self.port}, Fuel Amount: {self.fuel_amount}, Arrival Time: {self.arrival_time}, ' \
               f'Speed: {self.speed}, Price: {self.price}'

    def is_action_valid(self, action: BunkeringAction, dist_mat, route_schedule, fuel_capacity, k1, k2) -> bool:
        if action.refuel_amount + self.fuel_amount <= fuel_capacity:
            if not self.is_terminal:
                if action.next_port == route_schedule[self.port]:
                    if self.fuel_amount + action.refuel_amount >= fuel_consume_const_func(
                            dist_mat(self.port, route_schedule[self.port]), action.speed, k1, k2):
                        return True
        return False

    def resolve_action(self, action: BunkeringAction, dist_mat, route_schedule, arrival_time, fuel_capacity,
                       price_percentages, k1, k2) -> Optional[
        BunkeringState]:
        if self.is_action_valid(action, dist_mat, route_schedule, fuel_capacity, k1, k2):
            new_price_perc = self.price_perc * np.random.choice(price_percentages)
            return BunkeringState(port=route_schedule[self.port],
                                  fuel_amount=self.fuel_amount + action.refuel_amount - fuel_consume_const_func(
                                      dist_mat(self.port, route_schedule[self.port]), action.speed, k1, k2),
                                  speed=action.speed, arrival_time=arrival_time, price_perc=new_price_perc)

    def calculate_reward(self, refuel_amount):
        reward = 0
        reward -= refuel_amount * self.price
        reward -= self.fixed_bcost if refuel_amount > 0 else 0

        # SPEED AND ARRIVAL TIME CONSIDERATIONS ##
        if USE_SPEED:
            upper_arrival_time_dif = max(0, self.arrival_time - EXP_ARRIV_TIME_RNG[self.port][1])
            # lower_arrival_time_dif = max(0, EXP_ARRIV_TIME_RNG[self.port][0] - self.arrival_time)
            reward -= pow(upper_arrival_time_dif, KAPPA) * LATE_ARRIVAL_PENALTY
            # reward -= pow(lower_arrival_time_dif, KAPPA) * EARLY_ARRIVAL_PENALTY
        # SPEED AND ARRIVAL TIME CONSIDERATIONS ##

        # FUEL REMAINING CONSIDERATIONS ##
        # if self.fuel_amount < MIN_FUEL_ALLOWANCE:
        #   reward -= (MIN_FUEL_ALLOWANCE - self.fuel_amount) * KAPPA2
        reward = math.inf if self.fuel_amount + refuel_amount < 0 else reward

        return reward


class BunkeringMDP(MDP[BunkeringState, BunkeringAction]):
    def __init__(self, max_speed=MAX_SPEED, fuel_capacity=FUEL_CAPACITY, price_percentages=PRICE_PERCENTAGES,
                 state=BunkeringState(port=0, fuel_amount=0, speed=INITIAL_SPEED, arrival_time=0)):
        self.dist_mat = dist_matrix
        self.route_schedule = ROUTE_SCHEDULE[1:]
        self.min_fuel_allowance = MIN_FUEL_ALLOWANCE
        self.starting_state = state
        self.k1, self.k2 = k_coefficients(TEU)
        self.time = 0
        self.max_speed = max_speed
        self.fuel_capacity = fuel_capacity
        self.price_percentages = price_percentages
        self.min_set_size_speed = MIN_SET_SIZE_SPEED
        self.min_set_size_refuel = MIN_SET_SIZE_REFUEL
        self.min_set_size = MIN_SET_SIZE

    def is_terminal(self, state: BunkeringState) -> bool:
        return state.is_terminal

    def initial_state(self) -> BunkeringState:
        return self.starting_state

    def reward(self, current_state: Optional[BunkeringState], action: Optional[BunkeringAction]) -> float:
        return current_state.calculate_reward(action.refuel_amount)

    def transition(self, state: BunkeringState, action: BunkeringAction) -> BunkeringState:
        if state.is_terminal:
            return state
        travel_time = travel_time_function(self.dist_mat(state.port, self.route_schedule[state.port]), action.speed)
        target_state = state.resolve_action(action, self.dist_mat, self.route_schedule,
                                            state.departure_time + travel_time, self.fuel_capacity,
                                            self.price_percentages, self.k1, self.k2)
        return state if target_state is None else target_state

    # PROGRESSIVE WIDENING ALGORITHM #

    # def actions(self, state: BunkeringState, number_of_visits=None, iteration_number=None) -> list[BunkeringAction]:
    #     actions = list()
    #     if state.is_terminal:
    #         return actions
    #     possible_actions = self.get_possible_actions(state.fuel_amount, number_of_visits, iteration_number)
    #
    #     for a in possible_actions:
    #         if state.is_action_valid(BunkeringAction(a[0], a[1], self.route_schedule[state.port]), self.dist_mat,
    #                                  self.route_schedule):
    #             actions.append(BunkeringAction(a[0], a[1], self.route_schedule[state.port]))
    #     return actions

    def actions(self, state: BunkeringState, number_of_visits: int, iteration_number: int = None,
                max_iteration_number: int = None, dpw_exploration: dict = None, dpw_alpha: dict = None,
                min_action=False) -> list[
        TAction]:
        if os.environ['FORCE_0_FUEL'] == 'True':
            if state.port == ROUTE_SCHEDULE[-3]:  ## END UP WITH ZERO FUEL
                speed, distance = 1, DIST_MAT[state.port][self.route_schedule[state.port]]
                next_port= self.route_schedule[state.port]
                distance += DIST_MAT[next_port][self.route_schedule[next_port]]
                ref_amount = distance - state.fuel_amount if distance > state.fuel_amount else 0
                return [BunkeringAction(speed, ref_amount, self.route_schedule[state.port])]
            elif state.port == ROUTE_SCHEDULE[-2]:  ## END UP WITH ZERO FUEL
                speed, distance = 1, DIST_MAT[state.port][self.route_schedule[state.port]]
                ref_amount = distance - state.fuel_amount if distance > state.fuel_amount else 0
                return [BunkeringAction(speed, ref_amount, self.route_schedule[state.port])]
        actions = list()

        if state.is_terminal:
            return actions
        # possible_actions = self.get_possible_actions(state.fuel_amount, number_of_visits, iteration_number,
        #                                              max_iteration_number)
        if min_action and os.environ['ALGORITHM'] != 'NAIVE':
            possible_actions = self.get_all_actions(state.fuel_amount, number_of_visits, iteration_number,
                                                    max_iteration_number)

            return [BunkeringAction(1,min(a[1] for a in possible_actions), self.route_schedule[state.port])]


        if os.getenv('ALGORITHM') =='NAIVE':
            possible_actions = self.get_all_actions(state.fuel_amount, number_of_visits, iteration_number,
                                                         max_iteration_number)
        else:
            possible_actions = self.dpw_actions(number_of_visits, dpw_exploration, dpw_alpha)
        for a in possible_actions:
            if state.is_action_valid(BunkeringAction(a[0], a[1], self.route_schedule[state.port]), self.dist_mat,
                                     self.route_schedule, self.fuel_capacity, self.k1, self.k2):
                actions.append(BunkeringAction(a[0], a[1], self.route_schedule[state.port]))
        return actions

    def get_min_action(self, act):
        return act[1]

    # PROGRESSIVE WIDENING ALGORITHM #

    # def lookup(self, state: BunkeringState):
    #     policy_library = pd.read_csv('policies/policy_library.csv')
    #     try:
    #         refuel_amount = policy_library['refuel_amount'].to_numpy()[
    #             (policy_library['price'].to_numpy() == state.fue) & (
    #                     policy_library['port'].to_numpy() == sim_env.current_n) & (
    #                     policy_library['fuel_amount'].to_numpy() == sim_env.fuel_level)].item()

    def get_possible_actions(self, current_fuel, number_of_visits, iteration_number, max_iteration_number):

        if HEURISTIC:
            step_size_fuel, step_size_speed = 1, 1
        if iteration_number < max_iteration_number / 10:
            step_size_fuel, step_size_speed = 15, 15
        elif iteration_number < max_iteration_number / 5:
            if number_of_visits < iteration_number / 2:
                step_size_fuel, step_size_speed = 10, 8
            else:
                step_size_fuel, step_size_speed = 12, 12
        else:
            if number_of_visits < iteration_number / 5:
                step_size_fuel, step_size_speed = 15, 12
            elif number_of_visits < iteration_number / 2:
                step_size_fuel, step_size_speed = 25, 20
            else:
                step_size_fuel, step_size_speed = 50, 30

        possible_fuel_actions = set(np.linspace(0, self.fuel_capacity - current_fuel, num=step_size_fuel).astype(int))
        # possible_fuel_actions = set(np.linspace(0, self.fuel_capacity - current_fuel, num=int(self.fuel_capacity -
        # int(current_fuel)/4)).astype(int))
        # possible_speed_actions = set(np.linspace(1, self.max_speed, num=step_size_speed).astype(int))
        possible_speed_actions = [1]
        # possible_fuel_actions = [i for i in range(10,50)]
        possible_actions = [(x, y) for x in possible_speed_actions for y in possible_fuel_actions]
        return possible_actions

    def get_all_actions(self, current_fuel, number_of_visits, iteration_number, max_iteration_number):
        possible_fuel_actions = set(np.linspace(0, self.fuel_capacity, num=self.fuel_capacity).astype(int))
        possible_speed_actions = [1]
        possible_actions = [(x, y) for x in possible_speed_actions for y in possible_fuel_actions]
        return possible_actions

    # REDUNDANT
    def dpw_actions_dif(self, nVisits, dpw_exploration_refuel, dpw_alpha_refuel, dpw_exploration_speed, dpw_alpha_speed):
        possible_speed_actions = set(np.linspace(1, self.max_speed, num=int(self.min_set_size_speed)).astype(int))
        possible_fuel_actions = set(np.linspace(0, self.fuel_capacity, num=int(self.min_set_size_refuel)).astype(int))
        if nVisits == 0:
            return [(x, y) for x in possible_speed_actions for y in possible_fuel_actions]
        kSpeed = dpw_exploration_speed * (nVisits ** dpw_alpha_speed)
        set_size_speed = math.ceil(kSpeed)
        if set_size_speed <= self.min_set_size_speed:
            new_speed_actions = set(
                np.linspace(MIN_SPEED, self.max_speed, num=int(self.min_set_size_speed)).astype(np.uint64))
        elif set_size_speed < self.min_set_size_refuel * 2:
            possible_speed_actions = set(
                np.linspace(MIN_SPEED, self.fuel_capacity, num=int(self.min_set_size_speed)).astype(np.uint64))
            new = set(
                np.linspace(MIN_SPEED, self.fuel_capacity, num=int(self.min_set_size_speed * 2 - 1)).astype(np.uint64))
            dif = new.difference(possible_speed_actions)
            dif = list(dif)
            dif.sort()
            addition = dif[:int(set_size_speed - self.min_set_size_speed)]
            new_speed_actions = set(list(possible_speed_actions) + addition)
        else:
            self.min_set_size_speed = self.min_set_size_speed * 2 - 1
            new_speed_actions = set(
                np.linspace(MIN_SPEED, self.max_speed, num=int(self.min_set_size_speed * 2 - 1)).astype(np.uint64))

        kFuel = dpw_exploration_refuel * (nVisits ** dpw_alpha_refuel)
        set_size_fuel = math.ceil(kFuel)
        if set_size_fuel <= self.min_set_size_refuel:
            new_fuel_actions = set(
                np.linspace(0, self.fuel_capacity, num=int(self.min_set_size_refuel)).astype(np.uint64))
        elif set_size_fuel < self.min_set_size_refuel * 2:
            possible_fuel_actions = set(
                np.linspace(0, self.fuel_capacity, num=int(self.min_set_size_refuel)).astype(np.uint64))
            new = set(np.linspace(0, self.fuel_capacity, num=int(self.min_set_size_refuel * 2 - 1)).astype(np.uint64))
            dif = new.difference(possible_fuel_actions)
            dif = list(dif)
            dif.sort()
            addition = dif[:int(set_size_fuel - self.min_set_size_refuel)]
            new_fuel_actions = set(list(possible_fuel_actions) + addition)

            # print(f'IF TIME: {(time.time() - start) * 1000}')
        else:
            start = time.time()
            # print('MIN SET SIZE:' , self.min_set_size_refuel)
            self.min_set_size_refuel = self.min_set_size_refuel * 2 - 1
            new_fuel_actions = set(
                np.linspace(0, self.fuel_capacity, num=int(self.min_set_size_refuel * 2 - 1)).astype(np.uint64))

            # print(f'ELSE TIME: {(time.time() - start) * 1000}')

        new_actions = [(x, y) for x in new_speed_actions for y in new_fuel_actions]
        return new_actions

    def dpw_actions(self, nVisits, dpw_exploration, dpw_alpha):
        if USE_SPEED:
            possible_speed_actions = set(
                np.linspace(1, self.max_speed, num=int(self.min_set_size_speed)).astype(int))
            possible_fuel_actions = set(np.linspace(0, self.fuel_capacity, num=int(self.min_set_size_refuel)).astype(int))
            if nVisits == 0:
                return [(x, y) for x in possible_speed_actions for y in possible_fuel_actions]
            k = dpw_exploration * (nVisits ** dpw_alpha)
            set_size = math.ceil(k)
            if set_size <= self.min_set_size:
                return [(x, y) for x in possible_speed_actions for y in possible_fuel_actions]
            else :#set_size < min_set_size * 2:
                new = set(
                    np.linspace(1, self.max_speed, num=int(self.min_set_size_speed * 2 - 1)).astype(np.uint64))
                min_set_size_speed = self.min_set_size_speed * 2 - 1
                dif = new.difference(possible_speed_actions)
                dif = list(dif)
                dif.sort()
                addition_speed = dif[:int(set_size -self.min_set_size)]
                new = set(np.linspace(0, self.fuel_capacity, num=int(self.min_set_size_refuel * 2 - 1)).astype(np.uint64))
                min_set_size_refuel = self.min_set_size_refuel * 2 - 1
                dif = new.difference(possible_fuel_actions)
                dif = list(dif)
                dif.sort()
                addition_refuel = dif[:int(set_size - self.min_set_size)]
                t = len(possible_fuel_actions) * len(possible_speed_actions)
                i = 0
                while t < set_size:
                    t = (len(possible_fuel_actions) + 1) * (len(possible_speed_actions) + 1)
                    if t <set_size:
                        try:
                            possible_fuel_actions.add(addition_refuel[i])
                        except IndexError:
                            new = set(np.linspace(0, self.fuel_capacity, num=int(min_set_size_refuel * 2 - 1)).astype(np.uint64))
                            min_set_size_refuel = min_set_size_refuel * 2 - 1
                            dif = new.difference(possible_fuel_actions)
                            dif = list(dif)
                            dif.sort()
                            addition_refuel = dif[:int(set_size - self.min_set_size)]
                        try:
                            possible_speed_actions.add(addition_speed[i])
                        except:
                            new = set(
                                np.linspace(1, self.max_speed, num=int(min_set_size_speed * 2 - 1)).astype(np.uint64))
                            min_set_size_speed = min_set_size_speed * 2 - 1
                            dif = new.difference(possible_speed_actions)
                            dif = list(dif)
                            dif.sort()
                            addition_speed = dif[:int(set_size - self.min_set_size)]
                    elif (len(possible_fuel_actions) + 1) * (len(possible_speed_actions)) < set_size:
                        try:
                            possible_fuel_actions.add(addition_refuel[i])
                        except:
                            new = set(np.linspace(0, self.fuel_capacity, num=int(min_set_size_refuel * 2 - 1)).astype(np.uint64))
                            min_set_size_refuel = min_set_size_refuel * 2 - 1
                            dif = new.difference(possible_fuel_actions)
                            dif = list(dif)
                            dif.sort()
                            addition_refuel = dif[:int(set_size - self.min_set_size)]
                    i +=  1
                return [(x, y) for x in possible_fuel_actions for y in possible_speed_actions]
        else:
            set_size = math.ceil(dpw_exploration * pow(nVisits, dpw_alpha))
            if MIN_SET_SIZE_REFUEL >50:
                a=0
            if set_size < MIN_SET_SIZE_REFUEL:
                 set_size = MIN_SET_SIZE_REFUEL
            if set_size <= self.min_set_size_refuel:
                new_fuel_actions = set(
                    np.linspace(0, self.fuel_capacity, num=self.min_set_size_refuel).astype(int))
            elif set_size < self.min_set_size_refuel * 2:
                possible_fuel_actions = set(
                    np.linspace(0, self.fuel_capacity, num=self.min_set_size_refuel).astype(int))
                new = set(np.linspace(0, self.fuel_capacity, num=self.min_set_size_refuel * 2).astype(int))
                dif = new.difference(possible_fuel_actions)
                dif = list(dif)
                dif.sort()
                new_fuel_actions = list(possible_fuel_actions) + dif[:set_size - self.min_set_size_refuel]
            else:
                self.min_set_size_refuel = self.min_set_size_refuel * 2
                new_fuel_actions = list(np.linspace(0, self.fuel_capacity, num=self.min_set_size_refuel * 2).astype(int))

            new_speed_actions = [1]  ####### !!!
            new_fuel_actions = set(new_fuel_actions)
            new_actions = [(x, y) for x in new_speed_actions for y in new_fuel_actions]
            return new_actions