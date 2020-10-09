import sys, os
import json
import pickle
import shutil
from matplotlib import pyplot
from environments import *
from networks import *
from test_network import test_episodes

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
config_file = "project/experiments_config.json"
save_dir = "project/saved_experiments"

def load_config():
    def init_classes(config, module_name):
        # Convert class name strings to class instances
        for i in range(len(config[module_name])):
            module = __import__(module_name)
            class_name = config[module_name][i]
            class_ = getattr(module, class_name)
            config[module_name][i] = class_()

    with open(config_file) as f:
        config = json.load(f)

    init_classes(config, ENV_KEY)
    init_classes(config, NET_KEY)

    return config


def get_file_name(env,
                  net,
                  batch_size,
                  discount_factor,
                  semi_gradient,
                  layer,
                  seed):

    gradient_dir = 'semi' if semi_{{gradient else 'full'
    env_name = env.__class__.__name__
    net_name = net.__class__.__name__
    return f"{gradient_dir}/{env_name}/{net}_{batch_size}_{discount_factor}_{semi_gradient}_{layer}_{seed}"


def save_file(file, file_name):
    pickle.dump(f'project/{file_name}.pkl')


def run(env, net, batch_size, discount_factor, semi_gradient, layer):
    for seed_iter in range(num_runs):
        seed = seed_base + seed_iter

        file_name = get_file_name(env=env,
                                 net=net,
                                 batch_size=batch_size,
                                 discount_factor=discount_factor,
                                 semi_gradient=semi_gradient,
                                 layer=layer,
                                 seed=seed)
        
        if os.path.isfile(f'project/{file_name}.pkl'):
            continue

        random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)



        for epi in config[TRAIN_EPS_KEY]:
            pass

        for epi in config[TEST_EPS_KEY]:
            pass
    



def main(config):
    for env in config[ENV_KEY]:
        for net in config[NET_KEY]:
            for batch_size in config[BAT_KEY]:
                for discount_factor in config[DIS_KEY]:
                    for semi_gradient in config[GRA_KEY]:
                        for layer in config[LAYER_KEY]:
                            


                        # check if the files already exist

                        # run

                        # test for a number of episodes
                        pass
                        # save plots and data pickle files


if __name__ == "__main__":
    if overwrite_existing_files:
        answer = input(
            """
            Do you want to overwrite all the saved files?
            All the files will be deleted before starting.
            (y/n)   """)
        if answer == 'y':
            # will fail if any file is set to read-only
            shutil.rmtree(save_dir) 
            os.makedirs(save_dir)
        else:
            sys.exit(0)

    config = load_config()
    main(config)