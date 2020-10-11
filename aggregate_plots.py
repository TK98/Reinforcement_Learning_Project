from glob import glob

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import run_experiments as ex

def save_side_plot(plot_1, plot_1_name, plot_2, plot_2_name, file_name, title, extension='pdf'):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(plot_1)
    axes[0].set_title(plot_1_name)
    axes[1].plot(plot_2)
    axes[1].set_title(plot_2_name)
    fig.tight_layout()
    plt.title(title)
    plt.savefig(f'{ex.save_dir}/{file_name}.{extension}')
    plt.close(fig)


def aggregate(files, file_name, semi_gradient, env_name, net_name, smoothing=10):
    gradient_mode = 'Semi-gradient' if semi_gradient else 'Full-gradient'
    epi_dur_train = []
    losses = []

    epi_dur_test = []
    epi_r_test = []
    for file in files:
        with open(file, 'rb') as f:
            p = pickle.load(f)
            epi_dur_train.append(p['train']['episode_durations'])
            losses.append(p['train']['losses'])
            epi_dur_test.append(p['test']['episode_durations'])
            epi_r_test.append(p['test']['episode_rewards'])

    epi_dur_train = np.array(epi_dur_train).mean(axis=0)
    losses = np.array(losses).mean(axis=0)

    save_side_plot(ex.smooth(epi_dur_train, 10), 'Average pisode durations per episode',
                   ex.smooth(losses, 1), 'Average loss per episode',
                   file_name[:-1] + 'train_all',
                   f'Average {gradient_mode} {net_name} {env_name} Training')


    if np.array(epi_dur_test).shape[1] > 0: # There are test data
        epi_dur_test = np.array(epi_dur_test).mean(axis=0)
        epi_r_test = np.array(epi_r_test).mean(axis=0)
        save_side_plot(ex.smooth(epi_dur_test, 10), 'Episode durations per episode',
                       ex.smooth(epi_r_test, 1), 'Average loss per episode',
                       file_name[:-1] + 'test_all',
                       f'Average {gradient_mode} {net_name} {env_name} Test')

    print('Plot created.')


def get_files(env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size, lr_gamma, config):
    for num_episodes in config[ex.TRAIN_EPS_KEY]:
        file_name, _ = ex.get_file_name_and_config(env=env,
                                                   net=net,
                                                   batch_size=batch_size,
                                                   discount_factor=discount_factor,
                                                   semi_gradient=semi_gradient,
                                                   lr=lr,
                                                   lr_step_size=lr_step_size,
                                                   layer=layer,
                                                   lr_gamma=lr_gamma,
                                                   num_episodes=num_episodes,
                                                   seed="*")

        file_name = file_name.replace('[', ':left:').replace(']', '[]]').replace(':left:', '[[]')
        files = glob(f'{ex.save_dir}/{file_name}.pkl')

        return file_name, files


def do_stuff(env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size, lr_gamma, config):
    file_name, files = get_files(env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size, lr_gamma, config)

    if files:
        aggregate(files, file_name, semi_gradient, env.__name__, net.__name__)
    pass

def main(config):
    ex.do_loop(config, do_stuff)

if __name__ == "__main__":
    config_filename = 'experiments_config_cartpole.json'
    config = ex.load_config(config_filename)

    main(config)

    print('All plots aggregated.')