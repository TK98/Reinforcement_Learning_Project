import numpy as np
import torch

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