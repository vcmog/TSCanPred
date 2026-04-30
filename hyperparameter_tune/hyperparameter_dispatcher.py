import sys

sys.path.append("..")
import config as config
import json
import os

model_dir = config.model_dir


def get_hyperparameters(model_name):

    hyperparameter_path = model_dir + rf"\optimal_hyperparameters_{model_name}"
    if os.path.exists(hyperparameter_path):
        with open(hyperparameter_path, "r") as file:
            hyperparameters = json.load(file)
    else:
        hyperparameters = default_parameters[model_name]
    return hyperparameters


default_parameters = {
    "LSTM": {
        "lr": 0.001,
        #'batch_size': 64,
        "nhidden": 32,
        "n_layers": 1,
        "dropout": 0.2,
    },
    "LSTM_ALT": {
        "lr": 0.001,
        #'batch_size': 64,
        "nhidden": 32,
        "n_layers": 1,
        "dropout": 0.2,
    },
    "CNN": {
        "lr": 0.001,
        "batch_size": 64,
        "kernel_size": 3,
        "fcl_size": 32,
        "dropout": 0.2,
    },
    "LR": {
        "penalty": "l2",
        "tol": 0.0001,
        "C": 1,
        "class_weight": "balanced",
        "solver": "lbfgs",
    },
    "GRU": {
        "lr": 0.001,
        #'batch_size': 64,
        "nhidden": 32,
        "n_layers": 1,
        "dropout": 0.2,
        "n_power": 2,
    },
    "GRUD": {"nhidden": 33, "dropout": 0.25506139548130563},
    "Transformer": {"d_model":64, "nhead":2, "num_layers":2, "dim_feedforward":128, "dropout":0.1}
}
