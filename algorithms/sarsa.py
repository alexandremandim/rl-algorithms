# SARSA Algorithm
# On-Policy Algorith (Follows and Evaluate under the same policy)
# It's a TD Algorithm, so it uses bootstrapping (uses a guess of the next state value)
# Uses Gym Env to abstract Environment
# Tabular Version

import gym
import numpy as np
from gym.envs.registration import register
from algorithms.utils import policy_epsilon_greedy, NR_EPISODES, MAX_TIMESTEPS_PER_EPISODE, EPSILON_GREEDY, GAMMA, ALPHA


class Sarsa:
    def __init__(self, environment: gym.Env):
        self.env = environment
        self.Q = np.zeros((self.env.observation_space.nr_visits, self.env.action_space.nr_visits))

    def run_episode(self):
        # Action-Value Function Q(s,a)
        current_state = self.env.reset()
        current_action = policy_epsilon_greedy(self.Q, current_state, EPSILON_GREEDY)
        for _ in range(MAX_TIMESTEPS_PER_EPISODE):
            next_state, reward, done, info = self.env.step(current_action)
            next_action = policy_epsilon_greedy(self.Q, next_state, EPSILON_GREEDY)

            self.Q[current_state][current_action] += ALPHA * (reward + GAMMA * self.Q[next_state][next_action] - self.Q[current_state][current_action])

            current_state = next_state
            current_action = next_action

            if done:
                break


if __name__ == '__main__':
    register(id='MyWindyGridWorld-v0', entry_point='environments.windy_grid_world:WindyGridWorld')

    sarsa = Sarsa(gym.make('MyWindyGridWorld-v0'))
    for epiode in NR_EPISODES:
        sarsa.run_episode()
