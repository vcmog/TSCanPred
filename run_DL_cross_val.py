import argparse
import config
from torch.utils.data import Subset
from pathlib import Path
import os
import pandas as pd
from training.torch.training import train_model
from utils.io_utils import save_training_curve
from models import model_dispatcher
from hyperparameter_tune import hyperparameter_dispatcher
from datasets.sequence_datasets import Torch_Dataset
from datasets.preparation.common import (
    prepare_numpy_files_nosplit,
    initialise_dataloaders,
)
from datasets.collate import custom_collate_function
from evaluation.evaluation import evaluate_performance_torchmodel
from utils.data import count_outcomes_from_dataloader, compute_balanced_class_weights
from hyperparameter_tune.hyperparameter_tune import tune_torch_model_from_datasets
import utils.io_utils as gu
from utils.splits import get_kfold_split_indices
from utils.results import save_results_dfs, update_with_previous_fold_results, build_results_structures

if __name__ == "__main__":
    # initialise argument parser
    parser = argparse.ArgumentParser()

    # add arguments to parse
    parser.add_argument(
        "--model", type=str, choices=["LSTM", "CNN", "LSTM_ALT", "GRU", "Transformer"], required=True
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["RNN-sequenceclass", "RNN-lasthiddenstate", "CNN"],
        required=False,
        default = "RNN-lasthiddenstate"
    )
    parser.add_argument("--config", type=str, default=None, required=True)
    parser.add_argument("--lead_time", type=int, default=None, required=True)
    parser.add_argument("--n_epochs", type=int, default=10, required=False)
    parser.add_argument("--use_dems", dest="use_dems", action="store_true")
    parser.add_argument("--hyperparameter_tune", action="store_true")
    parser.add_argument("--no_dems", dest="use_dems", action="store_false")
    parser.add_argument("--earlystop_patience", type=int, default=5, required=False)
    parser.add_argument("--earlystop_delta", type=float, required=False, default=0.1)
    parser.add_argument("--max_seq_len", type=int, required=False, default=None)
    parser.add_argument("--min_seq_len", type=int, required=False, default=None)
    parser.add_argument("--start_fold", type=int, required=False, default=0, help="Fold to start on (0:5) (if training has been interrupted previously)")
    parser.set_defaults(use_dems=False, hyperparameter_tune=False)

    args = parser.parse_args()

model_name = args.model
model_type = args.model_type
lead_time = args.lead_time
n_epochs = args.n_epochs
use_dems = args.use_dems
hyperparameter_tune = args.hyperparameter_tune
earlystop_patience = args.earlystop_patience
earlystop_delta = args.earlystop_delta
max_seq_len = args.max_seq_len
min_seq_len = args.min_seq_len
config_file = args.config
start_fold = args.start_fold

if config_file is not None:
    project_dir, input_dir, output_dir, model_dir = config.get_dirs(config_file)

experiment_input_dir = input_dir + rf"\lead_time={lead_time}"
if not os.path.exists(experiment_input_dir):
    os.mkdir(experiment_input_dir)


if use_dems is True:
    static_str = "_with_static"
    print("Using static data.")
else:
    static_str = ""

model_dir, output_dir = gu.make_crossval_directories(
    lead_time, static_str, model_name=model_name, config_file=config_file
)

if not (
    Path(experiment_input_dir + rf"\{model_type}_inputs.npy").is_file()
    and Path(experiment_input_dir + rf"\{model_type}_labels.npy").is_file()
):

    n_features, seq_len, class_weights = prepare_numpy_files_nosplit(
        experiment_input_dir,
        experiment_input_dir,
        model_type,
        lead_time=lead_time,
        max_length=max_seq_len,
        min_length=min_seq_len,
    )
    # NOTE: max_length unused in LSTM_ALT and GRU
dataset = Torch_Dataset(
    experiment_input_dir + rf"\{model_type}_inputs.npy",
    experiment_input_dir + rf"\{model_type}_labels.npy",
    experiment_input_dir + rf"\{model_type}_ids.npy",
    experiment_input_dir + rf"\{model_type}_dems.npy",
)

best_params = hyperparameter_dispatcher.get_hyperparameters(model_name=model_name)
print(best_params)
lr = 0.0001  # best_params['lr'] # 0.0001  #
batch_size = 32  # 2 ** best_params['n_power'] #
for param in ["n_power", "lr"]:
    if param in best_params.keys():
        best_params.pop(param)

if model_type.startswith("RNN"):
    variable_dim = 2
elif model_type == "CNN":
    variable_dim = 1

splits = get_kfold_split_indices(dataset, [i[1] for i in dataset])

train_results, test_results, train_predictions, test_predictions = (
    build_results_structures()
)



update_with_previous_fold_results(model_name, start_fold, output_dir, static_str, train_results, test_results, train_predictions, test_predictions)

for fold, (fold_train, fold_test) in enumerate(splits):
    if fold < start_fold:
        continue
    train_set = Subset(dataset, fold_train)
    test_set = Subset(dataset, fold_test)

    train_loader, val_loader, test_loader, datasets = initialise_dataloaders(
        train_set,
        test_set,
        batch_size,
        variable_dim=variable_dim,
        split_train=True,
        collate_fn=custom_collate_function,
        return_datasets=True,
    )

    n_features = train_set[0][0].size(1)
    if hyperparameter_tune is True:
        train_set, val_set, test_set = datasets
        # tune_lstm_alt_from_datasets(train_set, val_set, n_trials=250, use_dems=use_dems)
        tune_torch_model_from_datasets(
            model_name,
            train_set,
            val_set,
            n_trials=250,
            use_dems=use_dems,
            n_features=n_features,
        )
        best_params = hyperparameter_dispatcher.get_hyperparameters(
            model_name=model_name
        )

    train_outcomes, train_len = count_outcomes_from_dataloader(train_loader)
    val_outcomes, val_len = count_outcomes_from_dataloader(val_loader)
    test_outcomes, test_len = count_outcomes_from_dataloader(test_loader)

    print("Train set:", train_len, train_outcomes, train_outcomes / train_len)
    print("Val set:", val_len, val_outcomes, val_outcomes / val_len)
    print("Test set:", test_len, test_outcomes, test_outcomes / test_len)

    first_sample_data = train_set[0][0]
    if model_type.startswith("RNN"):
        seq_len = first_sample_data.size(0)
        n_features = first_sample_data.size(1)
    elif model_type == "CNN":
        seq_len = first_sample_data.size(1)
        n_features = first_sample_data.size(0)

    if "class_weights" not in globals():
        y_train = dataset.labels[fold_train]
        class_weights = compute_balanced_class_weights(y_train)

    model = model_dispatcher.model_dispatcher[model_name](
        n_features=n_features, seq_len=seq_len, use_static=use_dems, **best_params
    )
    print("Model:", model)
    print("Training model...")
    train_losses, val_losses, best_val_loss = train_model(
        model,
        train_loader,
        val_loader,
        lr=lr,
        num_epochs=n_epochs,
        pos_class_weight=class_weights[0],
        save_dir=model_dir + rf"\lstm_lead_time={lead_time}",
        patience=earlystop_patience,
        min_earlystop_delta=earlystop_delta,
    )
    print("Model trained.")

    text = output_dir + rf"\{model_name}\cross-val{static_str}\{fold}_training_curve.pdf"
    print(repr(text))
    save_training_curve(
        train_losses,
        val_losses,
        output_dir + rf"\{model_name}\cross-val{static_str}\{fold}_training_curve.pdf",
    )

    print("Evaluating model...")

    fold_train_results, fold_train_predictions = evaluate_performance_torchmodel(
        model, train_loader, return_predictions=True
    )
    fold_test_results, fold_test_predictions = evaluate_performance_torchmodel(
        model, test_loader, return_predictions=True
    )

    fold_train_results["fold"] = fold_train_predictions["fold"] = fold_test_results[
        "fold"
    ] = fold_test_predictions["fold"] = fold

    print("Model evaluated.")
    print(f"Model performance on the test set for fold {fold}:  ", fold_test_results)
    train_results.loc[fold] = fold_train_results
    test_results.loc[fold] = fold_test_results
    train_predictions.append(fold_train_predictions)
    test_predictions.append(fold_test_predictions)
    gu.save_fold_results(model_name, output_dir, static_str, fold, fold_train_results, fold_train_predictions, fold_test_results, fold_test_predictions)
    print("Model evaluated.")
    print("Model performance on the train set:  ", train_results)
    print("Model performance on the test set:  ", test_results)

train_predictions = pd.concat(train_predictions)
test_predictions = pd.concat(test_predictions)


save_results_dfs(
    model_name=model_name,
    train_results=train_results,
    test_results=test_results,
    train_predictions=train_predictions,
    test_predictions=test_predictions,
    output_dir=output_dir,
    static_str=static_str
)
print(f"Done. Model {model} trained and evaluated at lead_time={lead_time} days.")
print(f"Results saved to: {output_dir}")
