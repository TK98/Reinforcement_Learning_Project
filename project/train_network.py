import torch
from torch import optim
import torch.nn.functional as F
import numpy as np

from .networks import Network, ReplayMemory

def get_epsilon(it):
    epsilon = 1.0 - 0.95 * it / 1000 if it < 1000 else 0.05
    return epsilon

def train_network(network, memory, optimizer, batch_size, semi_grad=True, use_replay=True):
    if use_replay:
        # don't learn without some decent experience
        if len(memory) < batch_size:
            return None

        # random transition batch is taken from experience replay memory
        transitions = memory.sample(batch_size)
    else:
        transitions = [memory.memory[-1]]

    parameters = network.memory_to_input(transitions)

    # compute the q value
    q_val = network.compute_q_vals(parameters['state'], parameters['action'])
    if semi_grad:
        with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
            target = network.compute_targets(**parameters)
    else:
        target = network.compute_targets(**parameters)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())


def train_episodes(env, policy, num_episodes, batch_size, learn_rate, semi_grad=True, use_replay=True):
    policy.train()
    network = policy.network

    memory = ReplayMemory(1000)
    optimizer = optim.Adam(network.parameters(), learn_rate)
    # policy = policy(network, 0.05)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []
    episode_rewards = []
    losses = []
    for i in range(num_episodes):

        network.start_episode(env, policy)

        steps = 0
        rewards = 0
        cum_loss = []
        while True:
            policy.update(global_steps)

            experience, done = network.step_episode(env, policy)

            memory.push(experience)
            loss = train_network(network, memory, optimizer, batch_size, semi_grad)
            if loss:
                cum_loss.append(loss)

            rewards += experience[2]
            global_steps += 1
            steps += 1

            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))

                episode_durations.append(steps)
                episode_rewards.append(rewards)
                mean_loss = np.mean(cum_loss) if cum_loss else 0
                losses.append(mean_loss)

                break

    return episode_durations, losses, episode_rewards