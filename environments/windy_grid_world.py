import sys

from gym import Env, spaces

GRID_WORLD_SIZE = 10 # 10 x 10

# https://rdrr.io/cran/reinforcelearn/man/WindyGridworld.html
class WindyGridWorld(Env):
    def __init__(self):
        self.action_space = spaces.Discrete(4) # up, down, left, right
        self.observation_space = spaces.Discrete(GRID_WORLD_SIZE * GRID_WORLD_SIZE)

        self.start_position = 30
        self.reward_position = 37

        self.current_position = self.start_position
        self.done = False

    def reset(self):
        self.done = False
        self.current_position = self.start_position
        return self.current_position

    def render(self, mode="human"):
        for row in range(GRID_WORLD_SIZE):
            for column in range(GRID_WORLD_SIZE):
                if self.current_position == row * 10 + column:
                    print('*\t', end='')
                elif row * 10 + column == self.reward_position:
                    print('G\t', end='')
                else:
                    print('.\t', end='')
            print('')
        print()
        print('0\t0\t0\t1\t1\t1\t2\t2\t1\t1\t')

    def step(self, action):
        reward = -1
        info = {}

        # up = -10, down = +10, right = +1, left = -1
        # Apply Action
        if action == 0: # UP
            # If we're in the top row, we cannot go up
            self.current_position = self.current_position - 10 if self.current_position > 9 else self.current_position
        elif action == 1: # DOWN
            # If we're in the booton row, we cannot go down
            self.current_position = self.current_position + 10 if self.current_position < 90 else self.current_position
        elif action == 2: # RIGHT
            # If we're in the rightmost column, cannot go right
            self.current_position = self.current_position + 1 if (self.current_position + 1) % 10 != 0 else self.current_position
        elif action == 3: # LEFT
            # If we're in the leftmost column, cannot go left
            self.current_position = self.current_position - 1 if self.current_position % 10 != 0 else self.current_position

        # Apply Wind
        # If we're in row 3, 4, 5, 8 -> +1 up
        if self.current_position in (list(range(3, 100, 10)) + list(range(4, 100, 10)) + list(range(5, 100, 10)) + list(range(8, 100, 10))):
            self.current_position = self.current_position if self.current_position in [3, 4 , 5 , 8] else self.current_position - 10
        # If we're in row 6, 7 -> +2 up
        elif self.current_position in (list(range(6, 100, 10)) + list(range(7, 100, 10))):
            if self.current_position in [6, 7]:
                pass
            elif self.current_position in [16, 17]:
                self.current_position -= 10

            else:
                self.current_position -= 20

        if self.current_position == self.reward_position:
            self.done
            reward = 0

        return self.current_position, reward, self.done, info


if __name__ == '__main__':
    print('Let''s Play')
    windy = WindyGridWorld()
    done = False

    windy.reset()
    windy.render()
    while not done:
        user_input = input('Play (w, a, s, d): ')[0]
        if user_input not in ['w', 's', 'd', 'a']:
            sys.exit(-1)
        if user_input == 'w':
            action = 0
        elif user_input == 's':
            action = 1
        elif user_input == 'd':
            action = 2
        elif user_input == 'a':
            action = 3

        new_state, reward, done, info = windy.step(action)
        print(f'Last Move: {user_input}\t Last Reward: {str(reward)}\tDone: {str(done)}')
        windy.render()
