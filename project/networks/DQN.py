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
#                                       Q networks                                          #
# ========================================================================================= #
class DeepQNetwork(Network):

    def __init__(self, in_features=4, out_features=2, architecture=None, discount_factor=0.8):
        nn.Module.__init__(self)

        if architecture is None:
            architecture = []

        nnet = []
        prev_layer = in_features
        for layer in architecture:
            nnet.append(nn.Linear(prev_layer, layer))
            nnet.append(nn.ReLU())
            prev_layer = layer
        nnet.append(nn.Linear(prev_layer, out_features))

        self.model = nn.Sequential(*nnet)

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

    def compute_targets(self, reward=None, next_state=None, done=None, **kwargs):
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
        next_q_vals = self.model.forward(next_state)
        next_max = torch.max(next_q_vals, dim=1).values.unsqueeze(1)

        if done.dtype != torch.uint8:
            adjusted_dones = torch.ones(done.size(), dtype=torch.uint8)
            adjusted_dones[done == True] = 0
        else:
            adjusted_dones = (done - 1) * -1  # Switches 0 and 1 around

        return reward + adjusted_dones * self.discount_factor * next_max

    def start_episode(self, env, policy):
        self.state = env.reset()

    def step_episode(self, env, policy):
        # Save old variables
        state = self.state

        # Get next state-action pair
        action = policy.sample_action(state)
        next_state, reward, done, _ = env.step(action)

        # Update state and action
        self.state = next_state

        return (state, action, reward, next_state, done), done

    @staticmethod
    def memory_to_input(transitions) -> Dict[str, torch.Tensor]:
        state, action, reward, next_state, done = zip(*transitions)

        # convert to PyTorch and define types
        parameters = dict()
        parameters['state'] = torch.tensor(state, dtype=torch.float, requires_grad=True)
        parameters['action'] = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
        parameters['reward'] = torch.tensor(reward, dtype=torch.float, requires_grad=True)[:, None]
        parameters['next_state'] = torch.tensor(next_state, dtype=torch.float, requires_grad=True)
        parameters['done'] = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean

        return parameters
