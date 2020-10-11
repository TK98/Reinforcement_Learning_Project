import argparse
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from aggregate_plots import get_files
import run_experiments as ex

config_filename = 'experiments_config_random.json'
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

def aggregate(files, env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size, lr_gamma):

    env_name = env.__name__
    file_name = f"{folder}/{'semi' if semi_gradient else 'full'}/{env_name}/qMSE_{net.__name__}_{batch_size}_{discount_factor}_{semi_gradient}_{lr}_{lr_step_size}_{lr_gamma}_{layer}_NUMEPISODES_*"
    
    qMSE_per_seed = []
    q_vals_per_seed = []
    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            qMSE = get_qMSE(data, env_name)

            non_terminal_last_q_vals = data['train']['q_vals'][-1][nonterminal_states[env_name]]

            qMSE_per_seed.append(qMSE)
            q_vals_per_seed.append(non_terminal_last_q_vals.numpy())
    
    mean_qMSE = np.mean(qMSE_per_seed, axis=0)
    plt.plot(mean_qMSE)
    plt.savefig(f'{file_name}.pdf')
    plt.close()

    # format error
    print(f'\n\nfor {file_name}:')
    print('State\tTrue\tComputed\tabs(Error):')
    table_rows = zip(np.mean(q_vals_per_seed, axis=0), true_nonterminal_q_vals[env_name])

    for i, (q_val, true_val) in enumerate(table_rows):
        print(f"{i}\t{true_val:.2f}\t{q_val:.2f}\t{abs(q_val-true_val):.2f}")

def do_stuff(env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size, lr_gamma, config):
    file_name, files = get_files(env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size, lr_gamma, config)
    aggregate(files, env, net, batch_size, discount_factor, semi_gradient, layer, lr, lr_step_size, lr_gamma)

ex.do_loop(ex.load_config(config_filename), do_stuff)