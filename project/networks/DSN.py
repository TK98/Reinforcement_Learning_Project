import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from typing import Dict

from .memory import ReplayMemory
from .network import Network


# ========================================================================================= #
#                                       SARSA networks                                      #
# ========================================================================================= #
class SARSANetwork(Network):

    def __init__(self, in_features=4, num_hidden=128, out_features=2, discount_factor=0.8):
        nn.Module.__init__(self)
        self.model = nn.Sequential(nn.Linear(in_features, num_hidden),
                                   nn.ReLU(),
                                   nn.Linear(num_hidden, out_features))

        self.discount_factor = discount_factor

    def forward(self, x):
        out = self.model.forward(x)
        return out

    def compute_q_vals(self, states, actions):
        """
        This method returns Q values for given state action pairs.

        Args:
            Q: Q-net
            states: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: Shape: batch_size x 1

        Returns:
            A torch tensor filled with Q values. Shape: batch_size x 1.
        """

        out = self.model.forward(states)
        return torch.gather(out, 1, actions)

    def compute_v_vals(self, states):
        pass

    def compute_targets(self, reward=None, next_state=None, next_action=None, done=None, **kwargs):
        """
        This method returns targets (values towards which Q-values should move).

        Args:
            Q: Q-net
            reward: a tensor of actions. Shape: Shape: batch_size x 1
            next_state: a tensor of states. Shape: batch_size x obs_dim
            done: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
            discount_factor: discount
        Returns:
            A torch tensor filled with target values. Shape: batch_size x 1.
        """
        next_q_vals = self.compute_q_vals(next_state, next_action)

        if done.dtype != torch.uint8:
            adjusted_dones = torch.ones(done.size(), dtype=torch.uint8)
            adjusted_dones[done == True] = 0
        else:
            adjusted_dones = (done - 1) * -1  # Switches 0 and 1 around

        return reward + adjusted_dones * self.discount_factor * next_q_vals

    def start_episode(self, env, policy):
        self.state = env.reset()
        self.action = policy.sample_action(self.state)

    def step_episode(self, env, policy):
        # Save old variables
        action = self.action
        state = self.state

        # Get next state-action pair
        next_state, reward, done, _ = env.step(action)
        next_action = policy.sample_action(next_state)

        # Update state and action
        self.state = next_state
        self.action = next_action

        return (state, action, reward, next_state, next_action, done), done

    @staticmethod
    def memory_to_input(transitions) -> Dict[str, torch.Tensor]:
        state, action, reward, next_state, next_action, done = zip(*transitions)

        # convert to PyTorch and define types
        parameters = dict()
        parameters['state'] = torch.tensor(state, dtype=torch.float, requires_grad=True)
        parameters['action'] = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
        parameters['reward'] = torch.tensor(reward, dtype=torch.float, requires_grad=True)[:, None]
        parameters['next_state'] = torch.tensor(next_state, dtype=torch.float, requires_grad=True)
        parameters['next_action'] = torch.tensor(next_action, dtype=torch.int64)[:, None]
        parameters['done'] = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean

        return parameters


# ========================================================================================= #
#                                       DEPRECATED                                          #
# ========================================================================================= #

# # ========================================================================================= #
# #                                   Utility functions                                       #
# # ========================================================================================= #
# def get_epsilon(it):
#     # YOUR CODE HERE
#     epsilon = 1.0 - 0.95 * it / 1000 if it < 1000 else 0.05
#     return epsilon


# class EpsilonGreedyPolicy(object):
#     """
#     A simple epsilon greedy policy.
#     """

#     def __init__(self, Q, epsilon):
#         self.Q = Q
#         self.epsilon = epsilon

#     def sample_action(self, obs):
#         """
#         This method takes a state as input and returns an action sampled from this policy.

#         Args:
#             obs: current state

#         Returns:
#             An action (int).
#         """
#         # YOUR CODE HERE
#         obs = torch.from_numpy(obs).float()
#         with torch.no_grad():
#             out = self.Q.forward(obs)

#         n_actions = len(out)
#         if np.random.rand() <= self.epsilon:
#             return np.random.randint(0, n_actions)
#         else:
#             return torch.argmax(out).item()

#     def set_epsilon(self, epsilon):
#         self.epsilon = epsilon

# # ========================================================================================= #
# #                                       Training                                            #
# # ========================================================================================= #
# def train_SARSANet(SARSANet, memory, optimizer, batch_size, semi_grad=True):
#     # DO NOT MODIFY THIS FUNCTION

#     # don't learn without some decent experience
#     if len(memory) < batch_size:
#         return None

#     # random transition batch is taken from experience replay memory
#     transitions = memory.sample(batch_size)

#     # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
#     state, action, reward, next_state, next_action, done = zip(*transitions)

#     # convert to PyTorch and define types
#     state = torch.tensor(state, dtype=torch.float)
#     action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
#     reward = torch.tensor(reward, dtype=torch.float)[:, None]
#     next_state = torch.tensor(next_state, dtype=torch.float)
#     next_action = torch.tensor(next_action, dtype=torch.int64)[:, None]
#     done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean

#     # compute the q value
#     q_val = SARSANet.compute_q_vals(state, action)
#     if semi_grad:
#         with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
#             target = SARSANet.compute_targets(reward, next_state, next_action, done)
#     else:
#         target = SARSANet.compute_targets(reward, next_state, next_action, done)

#     # loss is measured from error between current and newly expected Q values
#     loss = F.smooth_l1_loss(q_val, target)

#     # backpropagation of loss to Neural Network (PyTorch magic)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())


# def run_episodes(env, Q_network, num_episodes, batch_size, learn_rate, semi_grad=True):
#     memory = ReplayMemory(10000)
#     optimizer = optim.Adam(Q_network.parameters(), learn_rate)
#     policy = EpsilonGreedyPolicy(Q_network, 0.05)

#     global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
#     episode_durations = []  #
#     for i in range(num_episodes):

#         # Initial state action pair
#         state = env.reset()
#         with torch.no_grad():  # Is this needed?
#             action = policy.sample_action(state)

#         steps = 0
#         while True:
#             # YOUR CODE HERE
#             policy.set_epsilon(get_epsilon(global_steps))

#             next_state, reward, done, _ = env.step(action)

#             next_action = policy.sample_action(next_state)

#             memory.push((state, action, reward, next_state, next_action, done))
#             train_SARSANet(Q_network, memory, optimizer, batch_size, semi_grad)

#             state = next_state
#             action = next_action
#             global_steps += 1
#             steps += 1

#             if done:
#                 if i % 10 == 0:
#                     print("{2} Episode {0} finished after {1} steps"
#                           .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
#                 episode_durations.append(steps)
#                 break

#     return episode_durations
