import sys
import os
import json
import pickle
import shutil
import random
import time

import torch
import importlib
import matplotlib.pyplot as plt

import project
from project.environments import *
from project.networks import DeepQNetwork, DeepSARSA
from project.policies import EpsilonGreedyPolicy
from project.test_network import test_episodes
from project.train_network import train_episodes

# constants
ENV_KEY         = 'environments'
NET_KEY         = 'networks'
BAT_KEY         = 'batch-sizes'
DIS_KEY         = 'discount-factors'
GRA_KEY         = 'semi-gradient'
LAYER_KEY       = 'nn_layers'
TRAIN_EPS_KEY   = 'train-episodes'
TEST_EPS_KEY    = 'test-episodes'
LR_KEY          = 'lr'

# settings
seed_base = 42
num_runs = 10
overwrite_existing_files = False
config_file = "experiments_config.json"
save_dir = "saved_experiments"


def load_config():
    def init_classes(config, module_name):
        # Convert class name strings to class instances
        for i in range(len(config[module_name])):
            module = importlib.import_module(f'project.{module_name}')
            class_name = config[module_name][i]
            config[module_name][i] = getattr(module, class_name)

    with open(config_file) as f:
        config = json.load(f)

    init_classes(config, ENV_KEY)
    init_classes(config, NET_KEY)

    return config


def get_file_name_and_config(env,
                             net,
                             batch_size,
                             discount_factor,
                             semi_gradient,
                             lr,
                             layer,
                             seed):

    gradient_mode = 'semi' if semi_gradient else 'full'
    env_name = env.__name__
    net_name = net.__name__
    file_name = f'{gradient_mode}/{env_name}/{net_name}_{batch_size}_{discount_factor}_{semi_gradient}_{lr}_{layer}_{seed}'

    current_config = {
        ENV_KEY:    env_name,
        NET_KEY:    net_name,
        BAT_KEY:    batch_size,
        DIS_KEY:    discount_factor,
        GRA_KEY:    gradient_mode,
        LR_KEY:     lr,
        LAYER_KEY:  layer,
        seed:       seed
    }

    return file_name, current_config


def set_seeds(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)


def save_file(results, current_config, file_name):
    episode_durations_train, losses, episode_rewards_train, timespan, episode_durations_test, episode_rewards_test = results

    data = {
        "config": current_config,
        "train": {
            "episode_durations": episode_durations_train,
            "episode_rewards": episode_rewards_train,
            "losses": losses,
            "duration": timespan,
        },
        "test": {
            "episode_durations": episode_durations_test,
            "episode_rewards": episode_rewards_test
        }
    }
    pickle.dump(data, open(f'{save_dir}/{file_name}.pkl', 'wb'))


def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def save_plot(episode_durations, rewards, file_name, mode='train'):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(smooth(episode_durations, 10))
    axes[0].set_title('Episode durations per episode')
    axes[1].plot(smooth(rewards, 10))
    axes[1].set_title('Reward per episode')
    fig.tight_layout()
    plt.savefig(f'{save_dir}/{file_name}_{mode}.png')


def run(env, net, batch_size, discount_factor, semi_gradient, layer, lr, config):
    for seed_iter in range(num_runs):
        seed = seed_base + seed_iter

        file_name, current_config = get_file_name_and_config(env=env,
                                                             net=net,
                                                             batch_size=batch_size,
                                                             discount_factor=discount_factor,
                                                             semi_gradient=semi_gradient,
                                                             layer=layer,
                                                             lr=lr,
                                                             seed=seed)

        if os.path.isfile(f'{save_dir}/{file_name}.pkl'):
            continue

        os.makedirs(f'{save_dir}/{os.path.dirname(file_name)}', exist_ok=True)

        env_ins = env()

        print('Running: ', current_config)
        for epi in config[TRAIN_EPS_KEY]:
            # set the seeds in every iteration
            set_seeds(seed, env=env_ins)

            net_ins = net(in_features=env_ins.shape,
                          out_features=env_ins.action_space.n,
                          architecture=layer,
                          discount_factor=discount_factor)
            policy = EpsilonGreedyPolicy(net_ins)

            start = time.time()
            episode_durations_train, losses, episode_rewards_train = train_episodes(env=env_ins,
                                                                                    policy=policy,
                                                                                    num_episodes=epi,
                                                                                    batch_size=batch_size,
                                                                                    learn_rate=lr,
                                                                                    semi_grad=semi_gradient)

            timespan = time.time() - start
            print(f'Training finished in {timespan} seconds')

            torch.save(net_ins.state_dict(), f'{save_dir}/{file_name}.pt')

            # TODO: Q values (s,a) where #a=1 (it is V()) ,call compute_q_vals() NStep, Asplit every, in every episode, do not calculate the gradients
            # see how Q values evolve during training

            save_plot(episode_durations_train, losses, file_name)

            n_test_episodes = config[TEST_EPS_KEY]
            print(f'Start running {n_test_episodes} episodes for test')

            episode_durations_test, episode_rewards_test = test_episodes(env_ins, policy, n_test_episodes)
            save_plot(episode_durations_test,episode_rewards_test, file_name, mode='test')

            results = (episode_durations_train, losses, episode_rewards_train,
                       timespan, episode_durations_test, episode_rewards_test)
            save_file(results, current_config, file_name)


def main(config):
    for env in config[ENV_KEY]:
        for net in config[NET_KEY]:
            for batch_size in config[BAT_KEY]:
                for discount_factor in config[DIS_KEY]:
                    for semi_gradient in config[GRA_KEY]:
                        for layer in config[LAYER_KEY]:
                            for lr in config[LR_KEY]:
                                run(env=env,
                                    net=net,
                                    batch_size=batch_size,
                                    discount_factor=discount_factor,
                                    semi_gradient=semi_gradient,
                                    layer=layer,
                                    lr=lr,
                                    config=config)


if __name__ == "__main__":
    if overwrite_existing_files:
        answer = input(
            """
            Do you want to overwrite all the saved files?
            All the files will be deleted before starting.
            (y/n)   """)
        if answer == 'y':
            try:
                shutil.rmtree(save_dir)
            except:
                print('Error, perhaps some files are read-only.')

            os.makedirs(save_dir)
        else:
            sys.exit(0)

    config = load_config()
    main(config)
