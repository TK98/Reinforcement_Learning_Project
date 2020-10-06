from abc import ABC, abstractmethod

class Policy(ABC):
    """
    Policy abstract base class.
    """
    @abstractmethod
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        pass

    @property
    @abstractmethod
    def network(self):
        pass

    @abstractmethod
    def update(self, iteration):
        """Update policy parameters based on iteration count"""
        pass

    def train(self):
        """Set policy to training mode"""
        self.training = True

    def test(self):
        """Set policy to test mode"""
        self.training = False