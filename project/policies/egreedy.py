import numpy as np
import torch
from .policy import Policy

class EpsilonGreedyPolicy(Policy):
    """
    A simple epsilon greedy policy.
    """

    def __init__(self, Q, epsilon=1, start_e=1, end_e=0.05, steps_e=1000, training=True):
        self.Q = Q
        self.epsilon = epsilon
        self.start_e = start_e
        self.end_e = end_e
        self.steps_e = steps_e
        self.training = True

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
        if self.training and np.random.rand() <= self.epsilon:
            return np.random.randint(0, n_actions)
        else:
            return torch.argmax(out).item()

    @property
    def network(self):
        return self.Q

    def update(self, iteration):
        """Update policy parameters based on iteration count"""
        self.epsilon = self.get_epsilon(iteration, self.start_e, self.end_e, self.steps_e)

    @classmethod
    def get_epsilon(cls, it, start, end, steps):
        """Get value at it for linear anneal of an epsilon value from start to end over steps."""
        epsilon = start - (start - end) * it / steps if it < steps else end
        return epsilon