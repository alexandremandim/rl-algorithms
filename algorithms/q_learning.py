from gym import Env

from algorithms.utils import MAX_TIMESTEPS_PER_EPISODE, policy_epsilon_greedy, EPSILON_GREEDY, ALPHA, GAMMA, \
    policy_greedy


class QLearning:
    def __init__(self, environment: Env):
        self.environment = environment

    def run_episode(self):
        state = self.env.reset()

        for _ in range(MAX_TIMESTEPS_PER_EPISODE):
            action = policy_epsilon_greedy(self.Q, state, EPSILON_GREEDY)
            next_state, reward, done, info = self.env.step(action)

            self.Q[state][action] += ALPHA * (reward + GAMMA * policy_greedy(self.Q, state) - self.Q[state][action])

            state = next_state

            if done:
                break