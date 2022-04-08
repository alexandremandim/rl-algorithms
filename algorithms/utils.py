import random
import numpy as np


NR_EPISODES = 200
MAX_TIMESTEPS_PER_EPISODE = 1000
ALPHA = 0.9# aka Learning Rate. Influences how much we change Q(s,a) based on the new TD Target. 0, we never change the state. 1, we change to the maximum
GAMMA = 0.9# aka Discount Factor. Influences ao much we take into consideration the old value of Q(s,a). 0 -> Ignores value of next states. 1 -> Cares about future Q(s,a)
EPSILON_GREEDY = 0.8# Exploration paramater


def policy_epsilon_greedy(action_value_function, state, epsilon_greedy):
    if random.uniform(0, 1) <= epsilon_greedy:
        return policy_greedy(action_value_function, state)               # maximum action
    else:
        return np.random.choice(action_value_function.shape[1])           # random action


def policy_greedy(action_value_function, state):
    return my_argmax(action_value_function[state])


def my_argmax(array):
    indexes_list = []
    current_max = -np.inf
    for index, value in np.ndenumerate(array):
        if value > current_max:
            indexes_list = [index[0]]
            current_max = value
        elif value == current_max:
            indexes_list.append(index[0])

    return np.random.choice(indexes_list)
