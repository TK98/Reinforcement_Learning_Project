import argparse
import json
import pickle
import numpy as np
import seaborn as sns
from aggregate_plots import get_matched_files
import run_experiments as ex
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

config_filenames = ['experiments_config_asplit.json', 'experiments_config_random.json']
folder = 'saved_experiments'

nonterminal_states = {
    'ASplit' : [0,1,2],
    'NStateRandomWalk' : range(1,6)
}

true_nonterminal_q_vals = {
    'ASplit' : [0.5, 1, 0],
    'NStateRandomWalk' : [i/6 for i in range(1,6)]
}

def get_qMSE(data, env_name):
    q_vals_per_episode = data['train']['q_vals']
    qMSE = []
    for q_vals in q_vals_per_episode:
        nonterminal_q_vals = q_vals[nonterminal_states[env_name]]
        true_vals = true_nonterminal_q_vals[env_name]
        assert len(true_vals) == len(nonterminal_q_vals)

        qMSE.append(sum([(true_vals[i] - nonterminal_q_vals[i])**2 for i in range(len(true_vals))]))

    return [val.item() for val in qMSE]

def aggregate(files, env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size, lr_gamma, replay_memory):

    # we only plot DQN
    if net.__name__ == 'SARSANetwork':
        return

    env_name = env.__name__
    file_name = f"{folder}/{'semi' if semi_gradient else 'full'}/{env_name}/qMSE_{net.__name__}_{batch_size}_{discount_factor}_{semi_gradient}_{lr}_{lr_step_size}_{lr_gamma}_{layer}_{replay_memory}_NUMEPISODES_*"
    
    qMSE_per_seed = []
    q_vals_per_seed = []
    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            qMSE = get_qMSE(data, env_name)

            non_terminal_last_q_vals = data['train']['q_vals'][-1][nonterminal_states[env_name]]

            qMSE_per_seed.append(qMSE)
            q_vals_per_seed.append(non_terminal_last_q_vals.numpy())

    df = pd.DataFrame(qMSE_per_seed).T
    df.columns = ['semi-gradient' if semi_gradient else 'full-gradient'] * df.shape[1]
    
    sns.set_style(style='darkgrid')
    sns_plot = sns.lineplot(data=df, ci = 80, palette=['orange' if semi_gradient else 'blue'], dashes = False)
    leg_lines = sns_plot.legend().get_lines()
    for i in range(len(leg_lines)):
            marker = ':' if i % 2 == 0 else '-'
            sns_plot.lines[i].set_linestyle(marker)
            leg_lines[i].set_linestyle(marker)
    title = env_name
    sns_plot.set_title(f'qMSE over training for {title}')
    sns_plot.set_xlabel("Training episodes")
    sns_plot.set_ylabel(f'qMSE per episode')

def do_stuff(env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size, lr_gamma, replay_memory, config):
    files = get_matched_files(env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size, lr_gamma, replay_memory, config)
    aggregate(files, env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size, lr_gamma, replay_memory)
    

matplotlib.rcParams.update({'font.size': 13})

for config_filename in config_filenames:
    config = ex.load_config(config_filename)
    for _ in ex.do_loop(config, do_stuff):
        pass
    plt.show()
    plt.clf()