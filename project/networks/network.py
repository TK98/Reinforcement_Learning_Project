import torch
from torch import nn, optim
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict
import gym
from policies import Policy

class Network(ABC, nn.Module):
    @abstractmethod
    def compute_q_vals(self, states, actions):
        pass

    @abstractmethod
    def compute_v_vals(self, states):
        pass

    @abstractmethod
    def compute_targets(self, rewards, *args, **kwargs):
        pass

    @abstractmethod
    def start_episode(self, env: gym.Env, policy : Policy) -> None:
        pass

    @abstractmethod
    def step_episode(self, env: gym.Env, policy : Policy) -> Tuple[Tuple[Any, ...], bool]:
        pass

    @staticmethod
    @abstractmethod
    def memory_to_input(*args, **kwargs) -> Dict[str, torch.Tensor]:
        pass