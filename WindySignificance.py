from glob import glob

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import pandas as pd
import run_experiments as ex


def obtain_data(files, file_name, semi_gradient, env_name, net_name, smoothing=10):

    for file in files:
        with open(file, 'rb') as f:

            p = pickle.load(f)
            network = p['config']['networks']
            semi = p['config']['semi-gradient']
            test_duration = p['test']['episode_durations'][0]
            training_time = p['train']['duration']
            test_vals[network][semi].append(test_duration)
            time_vals[network][semi].append(training_time)



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
    file_name, files = get_files(env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size,
                                 lr_gamma, config)

    if files:
        obtain_data(files, file_name, semi_gradient, env.__name__, net.__name__)
    pass


def main(config):
    global test_vals
    global time_vals
    test_vals = {'SARSANetwork':
                     {'semi': [],
                      'full': []},
                 'DeepQNetwork':
                     {'semi': [],
                      'full': []}
                 }
    time_vals = {'SARSANetwork':
                     {'semi': [],
                      'full': []},
                 'DeepQNetwork':
                     {'semi': [],
                      'full': []}
                 }

    ex.do_loop(config, do_stuff)

    semi_sarsa_success = [v for v in test_vals['SARSANetwork']['semi'] if v <= 70]
    full_sarsa_success = [v for v in test_vals['SARSANetwork']['full'] if v <= 70]
    semi_dq_success = [v for v in test_vals['DeepQNetwork']['semi'] if v <= 70]
    full_dq_success = [v for v in test_vals['DeepQNetwork']['full'] if v <= 70]

    semi_sarsa_time = time_vals['SARSANetwork']['semi']
    full_sarsa_time = time_vals['SARSANetwork']['full']
    semi_dq_time = time_vals['DeepQNetwork']['semi']
    full_dq_time = time_vals['DeepQNetwork']['full']

    print(f"average successful semi-grad sarsa results: {np.mean(semi_sarsa_success):.3f}")
    print(f"average successful full-grad sarsa results: {np.mean(full_sarsa_success):.3f}")
    print(f"average successful semi-grad DQN results: {np.mean(semi_dq_success):.3f}")
    print(f"average successful full-grad DQN results: {np.mean(full_dq_success):.3f}")

    print("")

    print("semi sarsa vs full sarsa:")
    print(stats.ttest_ind(semi_sarsa_success, full_sarsa_success, equal_var=False))
    print("semi DQN vs full DQN:")
    print(stats.ttest_ind(semi_dq_success, full_dq_success, equal_var=False))

    print("")

    print(f"semi sarsa failed {100 - len(semi_sarsa_success)} times.")
    print(f"full sarsa failed {100 - len(full_sarsa_success)} times.")
    print(f"semi DQN failed {100 - len(semi_dq_success)} times.")
    print(f"full DQN failed {100 - len(full_dq_success)} times.")

    print("")

    print(f"semi sarsa took {np.mean(semi_sarsa_time):.3f} seconds to train")
    print(f"full sarsa took {np.mean(full_sarsa_time):.3f} seconds to train")
    print(f"semi DQN took {np.mean(semi_dq_time):.3f} seconds to train")
    print(f"full DQN took {np.mean(full_dq_time):.3f} seconds to train")



if __name__ == "__main__":
    config_filename = 'experiments_config.json'
    config = ex.load_config(config_filename)

    main(config)
