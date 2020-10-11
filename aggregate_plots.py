from glob import glob

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import run_experiments as ex

def aggregate(files, file_name):
    
    episode_durations = []
    episode_rewards = []
    for file in files:
        with open(file, 'rb') as f:
            p = pickle.load(f)
            pass

    pass



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
    file_name, files = get_files(env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size, lr_gamma, conf)
    aggregate(files, file_name)
    pass

def main(config):
    ex.do_loop(config, do_stuff)

if __name__ == "__main__":
    config_filename = 'experiments_config.json'
    config = ex.load_config(config_filename)

    main(config)