import random
import torch
import numpy as np

import matplotlib.pyplot as plt
import sys

assert sys.version_info[:3] >= (3, 6, 0), "Make sure you have Python 3.6 installed!"

import gym
env = gym.envs.make("CartPole-v1")

from project.networks import SARSANetwork, DeepQNetwork
from project.train_network import run_episodes
from project.policies import EpsilonGreedyPolicy

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

# Let's run it!
num_episodes = 100
batch_size = 64
discount_factor = 0.8
learn_rate = 1e-2
num_hidden = 128
seed = 42  # This is not randomly chosen

# We will seed the algorithm (before initializing QNetwork!) for reproducibility
set_seeds(seed)

my_DSN = DeepQNetwork(in_features=4, num_hidden=num_hidden, out_features=2, discount_factor=discount_factor)
episode_durations, losses = run_episodes(env, EpsilonGreedyPolicy, my_DSN, num_episodes, batch_size, learn_rate, semi_grad=True, use_replay=True)

# And see the results
def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

fig, axes = plt.subplots(nrows=1, ncols=2)

axes[0].plot(smooth(episode_durations, 10))
axes[0].set_title('Episode durations per episode')
axes[1].plot(smooth(losses, 10))
axes[1].set_title('Gradient loss per episode')
fig.tight_layout()
plt.show()