from glob import glob

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import run_experiments as ex

# def save_side_plot(plot_1, plot_1_name, plot_2, plot_2_name, file_name, title, extension='pdf'):
#     fig, axes = plt.subplots(nrows=1, ncols=2)
#     axes[0].plot(plot_1)
#     axes[0].set_title(plot_1_name)
#     axes[1].plot(plot_2)
#     axes[1].set_title(plot_2_name)
#     fig.tight_layout()
#     plt.title(title)
#     plt.savefig(f'{ex.save_dir}/{file_name}.{extension}')
#     plt.close(fig)


def plot(data_list):
    sns.set_style(style='darkgrid')
    data_tuples = [item for idx, item in enumerate(data_list)]

    envs = set([env for (env, _) in data_tuples]) # unique envs

    for env in envs:
        data = pd.concat([data for (_, data) in data_tuples], axis=1)

        sns_plot = sns.lineplot(data=data, color=sns.color_palette())

        sns_plot.set_title(f'{env}')
        sns_plot.set_xlabel("Episodes")
        sns_plot.set_ylabel("Episode duration per episode")
        # plt.savefig(f"{RESULTS_DIR}/lineplot_{group}.pdf", bbox_inches = 'tight', pad_inches = 0)
        plt.show()

        print('Plot created.')


def aggregate_data(files, file_name, semi_gradient, env_name, net_name):
    gradient_mode = 'Semi-gradient' if semi_gradient else 'Full-gradient'
    epi_dur_train = []
    # losses = []

    # epi_dur_test = []
    # epi_r_test = []
    for file in files:
        with open(file, 'rb') as f:
            p = pickle.load(f)
            epi_dur_train.append(p['train']['episode_durations'])
            # losses.append(p['train']['losses'])
            # epi_dur_test.append(p['test']['episode_durations'])
            # epi_r_test.append(p['test']['episode_rewards'])

    # epi_dur_train = np.array(epi_dur_train).mean(axis=0)
    df1 = pd.DataFrame(epi_dur_train).T
    df1.columns=[f'{gradient_mode} {net_name}' for i in range(df1.shape[1])]

    # df2 = pd.DataFrame(losses).T
    # df2.columns=['Losses' for i in range(df2.shape[1])]

    # df3 = pd.DataFrame(epi_dur_test).T
    # df3.columns=['Duration test episodes' for i in range(df3.shape[1])]

    # df4 = pd.DataFrame(epi_r_test)
    # df4.columns=['Test rewards' for i in range(df4.shape[1])]

    return df1


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
        return env.__name__, aggregate_data(files, file_name, semi_gradient, env.__name__, net.__name__)

def main(config):
    data_list = ex.do_loop(config, do_stuff)
    plot(data_list)


if __name__ == "__main__":
    config_filename = 'experiments_config_cartpole.json'
    config = ex.load_config(config_filename)

    main(config)

    print('All plots aggregated.')