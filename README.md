# On Full-Gradient Methods in Temporal-Difference Learning: An Empirical Study

This repository contains the source code for our paper "On Full-Gradient Methods in Temporal-Difference Learning: An Empirical Study", as of yet unpublished with no plans to publish this work in the future. We have looked at the difference in performance between taking the semi- and full-gradient in temporal difference control methods, namely SARSA and Q-learning. For both algorithms, we used non-linear function approximators in the form of neural networks.

In our repository, you will find an implementation for both SARSA and Q-learning, which can both be trained using either semi- or full-gradient descent methods. Furthermore, our figures can be replicated using the experimental code and config files in this repository.

## Folder structure
- `./project/` contains all code necessary to train and test one of the methods.
- `./README.md` is the file you are currently reading.
- `./aggregate_plots.py` is a file which contains the code to create the plots as seen in our paper.
- `./env.yml` conda environment used for this project.
- `./experiments_config_*.json` are files which contain the hyperparameters for different experiments.
- `./run_experiments.py` is the file which is used to fully run one of our experiments.
- `./show_q_vals.py` is used to compute the q-MSE per training step for ASplit and 7-Step random walk.

## Installation instructions
To run any of our code, install the conda environment as given in `./env.yml` using `conda env create --file env.yml`.
No further installation steps are required.

## Running the code
To train a network for an environment using our configurations, run `python run_experiments.py CONFIG_FILE`, where `CONFIG_FILE` is any of the `experiments_config_*.json` files.
