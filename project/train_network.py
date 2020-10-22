import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.optim import lr_scheduler

from .networks import ReplayMemory


def train_network(network, memory, optimizer, batch_size, semi_grad=True, use_replay=True):
    """

    :param network:
    :param memory:
    :param optimizer:
    :param batch_size:
    :param semi_grad:
    :param use_replay:
    :return:
    """
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

    # Returns a Python scalar, and releases history (similar to .detach())
    return loss.item()


def train_episodes(env, policy, num_episodes, batch_size, learn_rate, semi_grad, use_replay,
                   lr_step_size, lr_gamma, save_q_vals, replay_mem_size):
    """
    Trains a network on an environments for num_episodes episodes.

    :param env: The environment on which to train the network.
    :param policy: The policy to use. This policy must contain the policy network.
    :param num_episodes: For how many episodes to train.
    :param batch_size: The size of the training batches.
    :param learn_rate: The learning rate for Adam
    :param semi_grad: Whether to use semi-grad or full-grad.
    :param use_replay: Whether to use memory replay.
    :param lr_step_size: For the learning rate scheduler. Decay every lr_step_size episodes.
    :param lr_gamma: The scaling factor for the learning rate scheduler.
    :param save_q_vals: Whether to save the Q-values. If yes, saves all Q-values for all states after each episode.
    :param replay_mem_size: The size of the memory replay buffer.
    :return: The steps it took to complete each episode, the mean loss for each episode, the rewards per episode,
    and the Q-values of all states after each episode (only if save_q_vals is true).
    """

    policy.train()
    network = policy.network

    memory = ReplayMemory(replay_mem_size)
    optimizer = optim.Adam(network.parameters(), learn_rate)
    scheduler = lr_scheduler.StepLR(optimizer, lr_step_size, lr_gamma)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []
    episode_rewards = []
    losses = []
    q_vals = []
    for i in range(num_episodes):

        network.start_episode(env, policy)  # The network resets the environment and store the current state.

        steps = 0
        rewards = 0
        cum_loss = []
        while True:
            policy.update(global_steps)  # For the annealing of epsilon in the epsilon greedy policy.

            experience, done = network.step_episode(env, policy)  # Take a step.

            memory.push(experience)

            # Passes one training batch from the memory buffer through the network and update the weights once.
            loss = train_network(network, memory, optimizer, batch_size, semi_grad)
            if loss:
                cum_loss.append(loss)

            rewards += experience[2]
            global_steps += 1
            steps += 1

            if done:
                if len(memory) > batch_size and use_replay is True:
                    scheduler.step()
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))

                episode_durations.append(steps)
                episode_rewards.append(rewards)
                mean_loss = np.mean(cum_loss) if cum_loss else 0
                losses.append(mean_loss)

                if save_q_vals:
                    with torch.no_grad():
                        all_states = torch.eye(env.shape, dtype=torch.float)
                        q_val = network(all_states)
                        q_vals.append(q_val)

                break

    q_vals = torch.cat(q_vals, dim=1).T if save_q_vals else None

    return episode_durations, losses, episode_rewards, q_vals
