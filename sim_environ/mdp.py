from __future__ import annotations
import numpy as np
from common.properties import *
from typing import Optional
from sim_environ.cost_functions import *
from sim_environ.simulation_classes import *


class BunkerState():
    def __init__(self, x1: float, p: float, n: int, cum_f_cost: float):
        self.x1 = x1
        self.p = p
        self.n = n
        self.cum_f_cost = cum_f_cost


class BunkerMDP():
    def __init__(self, dist_mat, start_state, route_scehedule):

        self.dist_mat = dist_mat

        self.start_state = start_state
        self.current_state = self.start_state
        self.route_schedule = route_scehedule

    def initial_state(self) -> BunkerState:
        return self.start_state

    def is_terminal(self, state) -> bool:
        last_n = self.route_schedule[-1]
        return state.n == self.route_schedule[-1]

    def reward(self, previous_state: Optional[BunkerState], action: Optional[float], state: BunkerState) -> float:

        sim_env = MaritimeSim(dist_mat=self.dist_mat, ini_fuel_level=previous_state.x1, start_n=previous_state.n,
                              stoch_params=[STOCH_PROBS, STOCH_BUNKERING_COSTS])
        sim_env.cum_fuel_cost = previous_state.cum_f_cost
        cur_cost = previous_state.cum_f_cost
        sim_env.refuelAtPort(action, previous_state.n)
        added_cost = sim_env.cum_fuel_cost

        if cur_cost > 0:
            pass

        return -np.power(cur_cost - added_cost, 2)  # could make it state dependent

    def transition(self, state: BunkerState, action: float) -> BunkerState:
        if self.is_terminal(state):
            return state

        sim_env = MaritimeSim(dist_mat=self.dist_mat, ini_fuel_level=state.x1, start_n=state.n,
                              stoch_params=[STOCH_PROBS, STOCH_BUNKERING_COSTS])
        sim_env.cum_fuel_cost = state.cum_f_cost
        sim_env.refuelAtPort(action, state.n)
        n_index = self.route_schedule.index(sim_env.current_n)

        cost2 = sim_env.cum_fuel_cost

        if n_index + 1 < len(self.route_schedule):
            sim_env.traverse(sim_env.route_schedule[n_index + 1])

        return BunkerState(sim_env.fuel_level, sim_env.getFuelPrice(sim_env.current_n), sim_env.current_n,
                           sim_env.cum_fuel_cost)

    def actions(self, state: BunkerState, spacing=MCTS_DISCRETE_CARDINAL):
        sim_env = MaritimeSim(dist_mat=self.dist_mat, ini_fuel_level=state.x1, start_n=state.n,
                              stoch_params=[STOCH_PROBS, STOCH_BUNKERING_COSTS])
        action_ub = sim_env.fuel_capacity

        if self.route_schedule.index(state.n) + 1 < len(self.route_schedule):
            n_to = self.route_schedule[self.route_schedule.index(state.n) + 1]
        else:
            n_to = self.route_schedule[self.route_schedule.index(state.n)]

        fuel_consumed = sim_env.fuel_consume_func_callback(self.dist_mat[state.n][n_to])
        return list(np.linspace(fuel_consumed, action_ub, num=spacing))
