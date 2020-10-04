import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import mushroom_rl.features.basis as basis
import mushroom_rl.features.tensors.gaussian_tensor as gts
import mushroom_rl.features.tiles as tiles


# ========================================================================================= #
#                                   Utility functions                                       #
# ========================================================================================= #
def get_epsilon(it):
    # YOUR CODE HERE
    epsilon = 1.0 - 0.95 * it / 1000 if it < 1000 else 0.05
    return epsilon


class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """

    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon

    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        obs = torch.from_numpy(obs).float()
        with torch.no_grad():
            out = self.Q.forward(obs)

        n_actions = len(out)
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, n_actions)
        else:
            return torch.argmax(out).item()

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon


# ========================================================================================= #
#                                       Q networks                                          #
# ========================================================================================= #
class QNetwork(nn.Module):

    def __init__(self, discount_factor):
        nn.Module.__init__(self)
        self.model = None  # This must be defined in subclass as Sequential
        self.discount_factor = discount_factor

    def forward(self, x):
        out = self.model.forward(x)
        return out

    def compute_q_vals(self, state, action):
        # YOUR CODE HERE
        out = self.model.forward(state)
        return out[0][action]

    def compute_targets(self, reward, next_state, next_action, done):
        # YOUR CODE HERE
        next_q_val = self.compute_q_vals(next_state, next_action)
        done = int(done)
        done = (done - 1) * -1
        return reward + done * self.discount_factor * next_q_val


class DeepSARSA(QNetwork):

    def __init__(self, in_features=4, num_hidden=128, out_features=2, discount_factor=0.8):
        QNetwork.__init__(self, discount_factor)
        self.model = nn.Sequential(nn.Linear(in_features, num_hidden),
                                   nn.ReLU(),
                                   nn.Linear(num_hidden, out_features))
        # l1 = nn.Linear(in_features, num_hidden)
        # torch.nn.init.zeros_(l1.weight)
        # l1.bias.data.fill_(0.)
        # l2 = nn.Linear(num_hidden, out_features)
        # torch.nn.init.zeros_(l2.weight)
        # l2.bias.data.fill_(0.)
        # self.model = nn.Sequential(l1,
        #                            nn.ReLU(),
        #                            l2)


class LinearSARSA(QNetwork):

    def __init__(self, in_features=35, out_features=4, discount_factor=0.8):
        QNetwork.__init__(self, discount_factor)
        self.model = nn.Sequential(nn.Linear(in_features, out_features))

        # n_means = [35]
        # low = np.array([0])
        # high = np.array([70])



    # There is a github page out there with RBF. We can copy with citation right?

    def forward(self, x):
        # x = x.item()
        # features = [self.rbf[i](x) for i in range(len(self.rbf))]
        # features = torch.tensor(features, dtype=torch.float)
        out = self.model.forward(features)
        return out

    def compute_q_vals(self, state, action):
        # YOUR CODE HERE
        # state = state.item()
        # features = [self.rbf[i](state) for i in range(len(self.rbf))]
        # features = torch.tensor(features, dtype=torch.float)

        out = self.model.forward(features)
        return out[action]

# ========================================================================================= #
#                                       Training                                            #
# ========================================================================================= #
def update_network(state, action, reward, next_state, next_action, done,
                   Q_network, optimizer, semi_grad=True):

    state = torch.from_numpy(state).float()
    next_state = torch.from_numpy(next_state).float()

    q_val = Q_network.compute_q_vals(state, action)

    if semi_grad:
        with torch.no_grad():
            target = Q_network.compute_targets(reward, next_state, next_action, done)
    else:
        target = Q_network.compute_targets(reward, next_state, next_action, done)

    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def run_episodes(env, Q_network, num_episodes, learn_rate, semi_grad=True):
    optimizer = optim.Adam(Q_network.parameters(), learn_rate)
    policy = EpsilonGreedyPolicy(Q_network, 0.05)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        state = np.array([state])

        with torch.no_grad():  # Is this needed?
            action = policy.sample_action(state)

        steps = 0
        while True:
            # YOUR CODE HERE
            next_state, reward, done, _ = env.step(action)

            next_state = np.array([next_state])

            policy.set_epsilon(get_epsilon(int(global_steps/10)))
            next_action = policy.sample_action(next_state)

            update_network(state, action, reward, next_state, next_action, done, Q_network, optimizer, semi_grad)

            state = next_state
            action = next_action

            steps += 1
            global_steps += 1

            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                    episode_durations.append(steps)
                break

    print("asd")
    return episode_durations


# n_means = [35]
# low = np.array([0])
# high = np.array([70])
# a = gts.GaussianRBFTensor.generate(n_means, low, high)
#
#
# from windy_gridworld import WindyGridworldEnv
# env = WindyGridworldEnv()
#
# num_episodes = 100
# discount_factor = 0.8
# learn_rate = 1e-3
# num_hidden = 128
# seed = 42  # This is not randomly chosen
#
#
# my_DSARSA = LinearSARSA(discount_factor=discount_factor)
# episode_durations = run_episodes(env, my_DSARSA, num_episodes, learn_rate, semi_grad=True)

