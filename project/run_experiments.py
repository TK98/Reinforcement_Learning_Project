import json
import pickle
from environments import *
from networks import *
from test_network import test_episodes

env_key = 'environments'
net_key = 'networks'
bat_key = 'batch-sizes'
dis_key = 'discount-factors'
gra_key = 'semi-gradient'

def load_config():
    def init_classes(config, module_name):
        # Convert class name strings to class instances
        for i in range(len(config[module_name])):
            module = __import__(module_name)
            class_name = config[module_name][i]
            class_ = getattr(module, class_name)
            config[module_name][i] = class_()

    with open('config.json') as f:
        config = json.load(f)

    init_classes(config, env_key)
    init_classes(config, net_key)

    return config


def run(config):
    for env in config[env_key]:
        for net in config[net_key]:
            for batch_size in config[bat_key]:
                for discount_factor in config[dis_key]:
                    for semi_gradient in config[gra_key]:
                        # run
                        pass
                        # save plots and data pickle files


if __name__ == "__main__":    
    config = load_config()
    run(config)