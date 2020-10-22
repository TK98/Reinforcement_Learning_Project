from glob import glob

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import run_experiments as ex

plot_mode = ''
save_dir = ''


def plot(data_list):
    sns.set_style(style='darkgrid')
    data_tuples = [item for idx, item in enumerate(data_list)]

    envs = set([env for (env, _) in data_tuples]) # unique envs

    for env in envs:
        data = pd.concat([data for (data_env, data) in data_tuples if data_env == env],
                         axis=1)

        sns_plot = sns.lineplot(data=data, color=sns.color_palette(), ci=80)

        if plot_mode == 'time':
            print(data.mean().round(2))

        leg_lines = sns_plot.legend().get_lines()

        for i in range(len(leg_lines)):
            marker = ':' if i % 2 == 0 else '-'
            sns_plot.lines[i].set_linestyle(marker)
            leg_lines[i].set_linestyle(marker)

        title = input("Environment name for plot: ")

        sns_plot.set_title(f'Training {plot_mode} over episodes for {title}')
        sns_plot.set_xlabel("Episodes")
        sns_plot.set_ylabel(f'Training {plot_mode} per episode{" in seconds" if plot_mode == "time" else ""}')

        # sns_plot.set(ylim=(-100, 0))

        plt.show()
        plt.clf()

        print('Plot created.')


def aggregate_data(files, file_name, semi_gradient, env_name, net_name):
    gradient_mode = 'Semi-gradient' if semi_gradient else 'Full-gradient'
    data = []

    for file in files:
        with open(file, 'rb') as f:
            p = pickle.load(f)

            if plot_mode == "reward":
                data.append(p['train']['episode_rewards'])
            elif plot_mode == "time":
                 data.append([p['train']['duration']])

    df = pd.DataFrame(data)

    if plot_mode == "reward":
        df = df.T

    df.columns=[f'{gradient_mode} {net_name}' for i in range(df.shape[1])]

    return df


def get_files(env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size, lr_gamma, replay_memory, config):
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
                                                   replay_memory=replay_memory,
                                                   num_episodes=num_episodes,
                                                   seed="*")

        file_name = file_name.replace('[', ':left:').replace(']', '[]]').replace(':left:', '[[]')
        files = glob(f'{save_dir}/{file_name}.pkl')

        return file_name, files


def do_stuff(env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size, lr_gamma, rep_mem, config):
    file_name, files = get_files(env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size, lr_gamma, rep_mem, config)

    if files:
        return env.__name__, aggregate_data(files, file_name, semi_gradient, env.__name__, net.__name__)

def main(config):
    data_list = ex.do_loop(config, do_stuff)
    plot(data_list)


if __name__ == "__main__":
    config_filename = 'experiments_config_cartpole.json'
    save_dir = "saved_experiments_v2"
    config = ex.load_config(config_filename)

    plot_mode = "time" # reward or time

    main(config)

    print('All plots aggregated.')