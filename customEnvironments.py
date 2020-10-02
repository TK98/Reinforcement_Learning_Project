import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


# things to do for each env:
#   - step, reset and seed functions
#   - action_space, reward_range
# things I skipped:
#   - render function
#   - observation_space attribute
#   - close function

class BairdsCounterExample(gym.Env):
    action_space = spaces.Discrete(2)

    def __init__(self):
        self.n_states = 7
    
    def reset(self):
        self.current = np.random.choice(np.arange(0, 6))
        return self.current 
    
    def step(self, action):
        if action == 1:
            self.current = 6
        elif action == 0:
            self.current = np.random.choice(np.arange(0, 6))
        done = False
        return self.current, 0, done, ""

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class ASplit(gym.Env):
    action_space = spaces.Discrete(1)

    def __init__(self):
        # A,B,C + Terminal
        self.n_states = 4
    
    def reset(self):
        self.current = 0
        return self.current 
    
    def step(self, action):
        # there is only one action

        # if state is A
        if self.current == 0:
            # choose randomly between states 1 and 2
            self.current = np.random.choice([1, 2])
            done = False
            r = 0
        # if state is B
        elif self.current == 1:
            # go to terminal and give reward
            self.current = 3 
            done = True
            r = 1
        # if state is C
        elif self.current == 2:
            # same!
            self.current = 3
            done = True
            r = 0
        elif self.current == 3:
            raise Exception("You can't be in the terminal state. Perhaps you forgot to .reset()?")

        return self.current, r, done, ""

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class WindyGridWorld(gym.Env):
    action_space = spaces.Discrete(4) # u, d, r, l

    def __init__(self):
        # goal
        self.goal = (7, 3)
        # bounds
        self.x_l, self.x_u = 0, 9
        self.y_l, self.y_u = 0, 6
        # prepare wind
        self.wind = dict()
        for x in range(self.x_l, self.x_u+1):
            if x in [3,4,5,8]:
                self.wind[x] = 1
            elif x in [6,7]:
                self.wind[x] = 2
            else:
                self.wind[x] = 0
    
    def reset(self):
        self.x, self.y = 0, 3
        return (self.x, self.y)
    
    def step(self, action):
        # apply wind
        self.y += self.wind[self.x]
        self.y = min(self.y, self.y_u)
        
        # up
        if action == 0:
            self.y += 1
            self.y = min(self.y, self.y_u)
        # down
        elif action == 1:
            self.y -= 1
            self.y = max(self.y, self.y_l)
        # right
        elif action == 2:
            self.x += 1
            self.x = min(self.x, self.x_u)
        # left
        elif action == 3:
            self.x -= 1
            self.x = max(self.x, self.x_l)
        else:
            raise Exception("There is no such action!")

        # check goal!        
        done = (self.x, self.y) == self.goal

        return (self.x, self.y), -1, done, ""

    # No seed func. There is no randomness here!