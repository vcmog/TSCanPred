import argparse
import config
from models import dl_models, model_dispatcher
from hyperparameter_tune import hyperparameter_dispatcher
from pathlib import Path
from evaluation.evaluation import evaluate_performance_torchmodel
from utils import torch_model_utils
import json
import os
import torch
import numpy as np


if __name__ == "__main__":
    # initialise argument parser
    parser = argparse.ArgumentParser()

    # add arguments to parse
    parser.add_argument(
        "--model", type=str, choices=["LSTM", "CNN", "LSTM_ALT"], required=True
    )
    parser.add_argument("--model_type", type=str, choices=["RNN", "CNN"], required=True)
    parser.add_argument("--lead_time", type=int, default=None, required=True)
    parser.add_argument("--n_epochs", type=int, default=10, required=False)

    args = parser.parse_args()

model_name = args.model
model_type = args.model_type
lead_time = args.lead_time
n_epochs = args.n_epochs

input_dir = config.input_data_dir
experiment_input_dir = input_dir + rf"\lead_time={lead_time}"
if not os.path.exists(experiment_input_dir):
    os.mkdir(experiment_input_dir)

model_dir = config.model_dir + rf"\lead_time={lead_time}"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

output_dir = config.output_data_dir + rf"\lead_time={lead_time}"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    os.mkdir(output_dir + r"\train")
    os.mkdir(output_dir + r"\train\results")
    os.mkdir(output_dir + r"\train\predictions")
    os.mkdir(output_dir + r"\test")
    os.mkdir(output_dir + r"\test\results")
    os.mkdir(output_dir + r"\test\predictions")
# PRINT SOME STUFF HERE

if not (
    Path(input_dir + rf"\{model_name}_train_inputs.npy").is_file()
    and Path(input_dir + rf"\{model_name}_train_inputs.npy").is_file()
):
    n_features, seq_len, class_weights = torch_model_utils.prepare_numpy_files(
        input_dir, experiment_input_dir, model_type, lead_time=lead_time, max_length=10
    )

train_set = dl_models.Torch_Dataset(
    experiment_input_dir + rf"\{model_type}_train_inputs.npy",
    experiment_input_dir + rf"\{model_type}_train_labels.npy",
    experiment_input_dir + rf"\{model_type}_train_ids.npy",
)
test_set = dl_models.Torch_Dataset(
    experiment_input_dir + rf"\{model_type}_test_inputs.npy",
    experiment_input_dir + rf"\{model_type}_test_labels.npy",
    experiment_input_dir + rf"\{model_type}_test_ids.npy",
)


# tuned_hyperparam_path = config.model_dir + fr'\optimal_hyperparameters_{model_name}'
# with open(tuned_hyperparam_path, 'r') as f:
#  best_params=json.load(f)

best_params = hyperparameter_dispatcher.get_hyperparameters(model_name=model_name)
print(best_params)
lr = 0.00001  # best_params['lr'] # 0.007762950218670145
batch_size = 64  # 2 ** best_params['n_power'] #
for param in ["n_power"]:
    best_params.pop(param)


if model_type == "RNN":
    variable_dim = 2
elif model_type == "CNN":
    variable_dim = 1

train_loader, val_loader, test_loader = torch_model_utils.initialise_dataloaders(
    train_set, test_set, batch_size, variable_dim=variable_dim
)

train_outcomes = 0
train_len = 0
for _, labels, _ in train_loader:
    train_outcomes += sum(labels)
    train_len += len(labels)
print(train_outcomes)

val_outcomes = 0
val_len = 0
for _, labels, _ in val_loader:
    val_outcomes += sum(labels)
    val_len += len(labels)
print(val_outcomes)
print("Train set:", train_len, train_outcomes / train_len)
print("Val set:", val_len, val_outcomes / val_len)
if model_type == "RNN":
    seq_len = train_set[0][0].size(0)
    n_features = train_set[0][0].size(1)
elif model_type == "CNN":
    seq_len = train_set[0][0].size(1)
    n_features = train_set[0][0].size(0)

if not class_weights in globals():
    y_train = train_set.labels
    class_weights = torch.tensor(
        [len(y_train) / (2 * sum(y_train == 1)), len(y_train) / (2 * sum(y_train == 0))]
    )

# lstm = dl_models.MV_LSTM(n_features, seq_len, nhidden=nhidden, n_layers=n_layers, dropout=dropout)
model = model_dispatcher.model_dispatcher[model_name](
    n_features=n_features, seq_len=seq_len, **best_params
)
print("Model:", model)
print("Training model...")
train_losses, val_losses, best_val_loss = torch_model_utils.train_model(
    model,
    train_loader,
    val_loader,
    lr=lr,
    num_epochs=n_epochs,
    pos_class_weight=class_weights[0],
    save_dir=model_dir + rf"\lstm_lead_time={lead_time}",
)
print("Model trained.")

torch_model_utils.save_training_curve(
    train_losses, val_losses, output_dir + rf"\{model_name}_training_curve.pdf"
)

print("Evaluating model...")
if not os.path.exists(output_dir + r"\train"):
    os.mkdir(output_dir + r"\train")
train_results = evaluate_performance_torchmodel(
    model, train_loader, save_dir=output_dir + rf"\train\predictions\{model_name}"
)
if not os.path.exists(output_dir + r"\test"):
    os.mkdir(output_dir + r"\test")
test_results = evaluate_performance_torchmodel(
    model, test_loader, output_dir + rf"\test\predictions\{model_name}"
)
print("Model evaluated.")
print("Model performance on the train set:  ", train_results)
print("Model performance on the test set:  ", test_results)

## Change this to NP save (not json appropriate)
np.save(output_dir + rf"\train\results\{model_name}_train", train_results)

np.save(output_dir + rf"\test\results\{model_name}_test", test_results)
