import gym
import numpy as np
from gym import register

from algorithms.utils import MAX_TIMESTEPS_PER_EPISODE, policy_epsilon_greedy, EPSILON_GREEDY, GAMMA, ALPHA, NR_EPISODES


class SarsaLambda:
    def __init__(self, environment: gym.Env, lamba_value: float):
        if lamba_value < 0 or lamba_value > 1:
            raise Exception('Lambda must belong to [0, 1]')
        self.lamba_value = lamba_value
        self.env = environment
        self.Q = np.zeros((environment.observation_space.nr_visits, environment.action_space.nr_visits))

    def run_episode(self):
        eligibility_trace = np.zeros((self.env.observation_space.nr_visits, self.env.action_space.nr_visits))

        current_state = self.env.reset()
        current_action = policy_epsilon_greedy(self.Q, current_state, EPSILON_GREEDY)

        for _ in range(MAX_TIMESTEPS_PER_EPISODE):
            next_state, reward, done, info = self.env.step(current_action)
            next_action = policy_epsilon_greedy(self.Q, current_state, EPSILON_GREEDY)

            td_error = reward + GAMMA * self.Q[next_state, next_action] - self.Q[current_state][current_action]
            eligibility_trace[current_state][current_action] += 1

            # Update all eligibity values to all (state, actions)
            for state in range(self.env.observation_space.nr_visits):
                for action in range(self.env.action_space.nr_visits):
                    self.Q[state][action] += ALPHA * td_error * eligibility_trace[state][action]
                    eligibility_trace[state][action] *= GAMMA * self.lamba_value

            current_state = next_state
            current_action = next_action

            if done:
                break


if __name__ == '__main__':
    register(id='MyWindyGridWorld-v0', entry_point='environments.windy_grid_world:WindyGridWorld')

    sarsa = SarsaLambda(gym.make('MyWindyGridWorld-v0'))
    for epiode in NR_EPISODES:
        sarsa.run_episode()