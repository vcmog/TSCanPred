import config
from models import dl_models
import optuna
import json
import os
import gc
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from plotly.io import show
from models.dl_models import GRUD
from models.dl_models import EarlyStopper
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from training.torch.set_up import set_up_training_components
from utils.data import compute_balanced_class_weights
from models.model_dispatcher import model_dispatcher
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
from datasets.sequence_datasets import Torch_Dataset
from datasets.utils.utils import get_loader_with_batch_sampling
from datasets.samplers import BatchSampler
from datasets.collate import custom_collate_function, custom_collate_function_grud
from datasets.preparation.common import (
    prepare_numpy_files_with_split,
    initialise_dataloaders,
)
from training.torch.training import run_1_epoch
from sklearn.neural_network import MLPClassifier

# Hyperparameter tuning is performed at lead_time = 0
model_dir = config.model_dir
input_dir = config.input_data_dir
experiment_input_dir = input_dir + rf"\lead_time={0}"
if not os.path.exists(experiment_input_dir):
    os.mkdir(experiment_input_dir)


def hyperparameter_tune_sklearn_model(model_name, X, y, n_trials):

    X_dev, X_val, y_dev, y_val = train_test_split(X, y, test_size=0.25)
    model_cfg = MODEL_REGISTRY[model_name]
    model_builder = model_cfg["builder"]
    model_params = model_cfg["param_func"]
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    def objective(trial):

        params = model_params(trial)
        model = model_builder(**params)
        model.fit(X_dev, y_dev)

        y_probs = model.predict_proba(X_val)[:, 1].ravel()
        y_pred = y_probs > 0.5
        # balanced_accuracy = balanced_accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred)

        return auc

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.HyperbandPruner
    )
    probs = study.optimize(objective, n_trials=n_trials, gc_after_trial=True, n_jobs=4)
    print("Best hyperparameters:", study.best_params)
    print("Best auc:", study.best_value)
    best_params = study.best_params

    if "penalty_solver_pair" in best_params.keys():
        best_params["penalty"] = best_params["penalty_solver_pair"].split(",")[0]
        best_params["solver"] = best_params["penalty_solver_pair"].split(",")[1]
        del best_params["penalty_solver_pair"]

    file_name = model_dir + rf"\optimal_hyperparameters_{model_name}"
    with open(file_name, "w") as f:
        json.dump(study.best_params, f)
    return best_params


def tune_cnn(n_trials):

    n_features, seq_len, class_weights = prepare_numpy_files_with_split(
        input_dir,
        experiment_input_dir=experiment_input_dir,
        model="CNN",
        lead_time=0,
        max_length=20,
    )

    train_set = Torch_Dataset(
        experiment_input_dir + rf"\CNN_train_inputs.npy",
        experiment_input_dir + rf"\CNN_train_labels.npy",
        experiment_input_dir + rf"\CNN_train_ids.npy",
    )
    test_set = Torch_Dataset(
        experiment_input_dir + rf"\CNN_test_inputs.npy",
        experiment_input_dir + rf"\CNN_test_labels.npy",
        experiment_input_dir + rf"\CNN_test_ids.npy",
    )

    early_stopper = EarlyStopper(patience=5)

    def objective(trial):
        lr = 0.001  # trial.suggest_float('lr', 1e-8, 1e-1, log=True)
        # n = trial.suggest_int('n_power', 2, 12) # 16 to 4096
        batch_size = 64  # 2**n
        kernel_size = trial.suggest_int("kernel_size", 1, 4)
        fcl_size = trial.suggest_int("fcl_size", 1, 100)
        dropout = trial.suggest_float("dropout", 0.1, 0.9)
        n_epochs = 50

        train_loader, val_loader, _ = initialise_dataloaders(
            train_set, test_set, batch_size, variable_dim=1, train_val_split=0.5
        )
        y_train = train_set.labels
        class_weights = torch.tensor(
            [
                len(y_train) / (2 * sum(y_train == 1)),
                len(y_train) / (2 * sum(y_train == 0)),
            ]
        )

        seq_len = train_set[0][0].size(1)
        n_features = train_set[0][0].size(0)

        model = dl_models.onedCNN(
            n_features=n_features,
            seq_len=seq_len,
            kernel_size=kernel_size,
            fcl_size=fcl_size,
            dropout=dropout,
        )
        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[0], reduction="mean")
        unweighted_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        for epoch in range(n_epochs):
            model.train()
            # Create a progress bar
            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False
            )
            running_loss = 0.0

            # Iterate over the dataset
            for batch in progress_bar:
                inputs = batch[0]
                labels = batch[1].flatten()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Update the running loss
            running_loss += loss.item()

            # Update the progress bar with the current loss
            progress_bar.set_postfix({"loss": running_loss / len(progress_bar)})

            # Calculate average loss for the epoch
            # epoch_loss = running_loss / len(train_loader)
            # training_losses.append(epoch_loss)

            # Validation phase
            model.eval()
            running_val_loss = 0.0

            # calculate epoch val los and report intermediate balanced accuracy to the trial
            predicted_labels = []
            true_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch[0]
                    labels = batch[1].flatten()
                    outputs = model(inputs)
                    loss = unweighted_criterion(outputs, labels)
                    running_val_loss += loss.item()

                    probabilities = torch.sigmoid(outputs)
                    predicted = (probabilities > 0.5).int()
                    predicted_labels.extend(predicted.numpy())
                    true_labels.extend(labels.numpy())

            inter_balanced_accuracy = balanced_accuracy_score(
                true_labels, predicted_labels
            )
            trial.report(inter_balanced_accuracy, epoch)
            # Calculate average validation loss for the epoch
            epoch_val_loss = running_val_loss / len(val_loader)
            # val_losses.append(epoch_val_loss)
            if early_stopper.early_stop(epoch_val_loss):
                print(f"Early stopping triggered. Stopping at epoch {epoch+1}")
                break
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        # End of training loop #

        # report balanced accuracy on validation set
        predicted_labels = []
        true_labels = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0]
                labels = batch[1].flatten()
                outputs = model(inputs)
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).int()
                predicted_labels.extend(predicted.numpy())
                true_labels.extend(labels.numpy())
        balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)
        return balanced_accuracy

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("Best hyperparameters:", study.best_params)
    print("Best val_loss:", study.best_value)

    file_name = model_dir + rf"\optimal_hyperparameters_CNN"

    with open(file_name, "w") as f:
        json.dump(study.best_params, f)

    fig = optuna.visualization.plot_optimization_history(study)
    show(fig)


def tune_lstm_alt_from_datasets(train_set, val_set, n_trials, use_dems=False):
    """Tune LSTM hyperparameters using the Optuna library. Uses val set to evaluate model, and splits
    train set to make an early stopping validation set."""

    def objective(trial):
        # n = trial.suggest_int('n_power', 4, 12) # 16 to 4096
        batch_size = 32  # 2**n
        train_loader, early_stopping_loader, val_loader = initialise_dataloaders(
            train_set,
            val_set,
            batch_size,
            variable_dim=2,
            train_val_split=0.5,
            collate_fn=custom_collate_function,
        )
        lr = 0.0001  # trial.suggest_float('lr', 1e-6, 1e-1, log=True)

        nhidden = trial.suggest_int("nhidden", 16, 64)
        n_layers = trial.suggest_int("n_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.9)
        y_train = torch.tensor([sample[1][0] for sample in train_set])
        class_weights = compute_balanced_class_weights(y_train)
        seq_len = train_set[0][0].size(0)
        n_features = train_set[0][0].size(1)
        model = dl_models.Hiddenstate_LSTM(
            n_features=n_features,
            #seq_length=seq_len,
            nhidden=nhidden,
            n_layers=n_layers,
            dropout=dropout,
            use_static=use_dems,
        )
        # Define loss function and optimizer
        criterion, optimizer, scheduler, early_stopper = set_up_training_components(
            model=model, lr=lr, class_weights=class_weights
        )
        n_epochs = 100

        for epoch in tqdm(range(n_epochs)):
            # Train
            epoch_train_loss = run_1_epoch(
                model, train_loader, criterion, optimizer, train=True
            )
            # Evaluate on early stopping set and report intermediate balanced_accuracy
            epoch_es_loss, epoch_es_bal_acc = run_1_epoch(
                model,
                early_stopping_loader,
                criterion,
                train=False,
                additional_metric=balanced_accuracy_score,
            )
            scheduler.step(epoch_es_loss)
            trial.report(epoch_es_bal_acc, epoch)

            if early_stopper.early_stop(epoch_es_loss):
                print(f"Early stopping triggered. Stopping at epoch {epoch+1}")
                break
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            gc.collect()
            tqdm.write(
                f"Epoch {epoch+1}/{n_epochs} |"
                f"Train loss: {epoch_train_loss:.4f} | Early stopping loss: {epoch_es_loss:.4f} "
            )

        val_loss, val_bal_acc = run_1_epoch(
            model,
            val_loader,
            criterion,
            train=False,
            additional_metric=balanced_accuracy_score,
        )
        del model
        del optimizer
        gc.collect()
        return val_bal_acc

    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=15)
    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print("Best hyperparameters:", study.best_params)
    print("Best balanced accuracy", study.best_value)

    file_name = model_dir + r"\optimal_hyperparameters_LSTM_ALT"

    with open(file_name, "w") as f:
        json.dump(study.best_params, f)

    fig = optuna.visualization.plot_optimization_history(study)
    show(fig)


def tune_GRUD_from_datasets(dev_set, val_set, n_trials, X_mean):
    """Tune LSTM hyperparameters using the Optuna library. Uses val set to evaluate model, and splits
    train set to make an early stopping validation set."""
    batch_size = 64

    train_set, early_stopping_set = torch.utils.data.random_split(dev_set, [0.8, 0.2])
    train_loader = get_loader_with_batch_sampling(
        train_set, batch_size=batch_size, num_bins=2
    )

    early_stopping_lengths = [sample[3] for sample in early_stopping_set]
    early_stopping_batches = BatchSampler(
        early_stopping_lengths, batch_size=batch_size, num_bins=2
    )
    early_stopping_loader = DataLoader(
        val_set,
        batch_sampler=early_stopping_batches,
        collate_fn=custom_collate_function,
    )

    val_loader = DataLoader(val_set, shuffle=False, batch_size=1)

    def objective(trial):
        lr = 0.0001  # trial.suggest_float('lr', 1e-6, 1e-1, log=True)
        # n = trial.suggest_int('n_power', 4, 12) # 16 to 4096
        # 2**n
        nhidden = trial.suggest_int("nhidden", 16, 64)
        dropout = trial.suggest_float("dropout", 0.1, 0.9)

        y_train = torch.tensor([sample[1][0] for sample in train_set])
        class_weights = compute_balanced_class_weights(y_train)

        n_features = X_mean.shape[-1]

        model = GRUD.GRUD(
            input_size=n_features, nhidden=nhidden, X_mean=X_mean, dropout=dropout
        )
        criterion, optimizer, scheduler, early_stopper = set_up_training_components(
            model, lr, class_weights
        )
        n_epochs = 100

        for epoch in tqdm(range(n_epochs), desc="Epochs"):
            # Train
            epoch_train_loss = run_1_epoch(
                model, train_loader, criterion, optimizer, train=True
            )
            # Evaluate on early stopping set and report intermediate balanced_accuracy
            epoch_es_loss, epoch_es_bal_acc = run_1_epoch(
                model,
                early_stopping_loader,
                criterion,
                train=False,
                additional_metric=balanced_accuracy_score,
            )
            scheduler.step(epoch_es_loss)
            trial.report(epoch_es_bal_acc, epoch)

            if early_stopper.early_stop(epoch_es_loss):
                print(f"Early stopping triggered. Stopping at epoch {epoch+1}")
                break
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            gc.collect()
            tqdm.write(
                f"Epoch {epoch+1}/{n_epochs} |"
                f"Train loss: {epoch_train_loss:.4f} | Early stopping loss: {epoch_es_loss:.4f} |"
                f" Learning Rate: {scheduler.get_last_lr()[0]}"
            )

        val_loss, val_bal_acc = run_1_epoch(
            model,
            val_loader,
            criterion,
            train=False,
            additional_metric=balanced_accuracy_score,
        )
        del model
        del optimizer
        gc.collect()
        return val_bal_acc

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, n_jobs=3)

    print("Best hyperparameters:", study.best_params)
    print("Best balanced accuracy", study.best_value)

    file_name = model_dir + rf"\optimal_hyperparameters_GRU_D"

    with open(file_name, "w") as f:
        json.dump(study.best_params, f)

    fig = optuna.visualization.plot_optimization_history(study)
    show(fig)


def tune_torch_model_from_datasets(
    model_name, dev_set, val_set, n_trials, use_dems, n_features, **builder_kwargs
):
    """Tune LSTM hyperparameters using the Optuna library. Uses val set to evaluate model, and splits
    train set to make an early stopping validation set.
     For GRUD Xmean is an important kwarg"""
    cfg_model = MODEL_REGISTRY[model_name]
    model_builder = cfg_model["builder"]
    param_func = cfg_model["param_func"]
    collate_fn = cfg_model["collate_fn"]
    batch_size = 64
    train_set, early_stopping_set = torch.utils.data.random_split(dev_set, [0.8, 0.2])
    train_loader = get_loader_with_batch_sampling(
        train_set, batch_size=batch_size, num_bins=2, collate_fn=collate_fn
    )
    early_stopping_loader = get_loader_with_batch_sampling(
        early_stopping_set, batch_size=batch_size, num_bins=2, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_set, shuffle=False, batch_size=1)

    data = [train_loader, early_stopping_loader, val_loader]

    study = optuna.create_study(direction="maximize")
    objective = make_objective(
        model_builder=model_builder,
        param_func=param_func,
        data=data,
        train_set=train_set,
        use_dems=use_dems,
        n_features=n_features,
        **builder_kwargs,
    )

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, n_jobs=3)

    print("Best hyperparameters:", study.best_params)
    print("Best auroc", study.best_value)

    file_name = model_dir + rf"\optimal_hyperparameters_{model_name}"

    with open(file_name, "w") as f:
        json.dump(study.best_params, f)

    fig = optuna.visualization.plot_optimization_history(study)
    show(fig)


def make_objective(
    model_builder,
    param_func,
    data,
    train_set,
    n_features,
    use_dems,
    epochs=100,
    **builder_kwargs,
):
    def objective(trial):
        params = param_func(trial)
        lr = params.pop("lr")
        model = model_builder(
            n_features=n_features, use_static=use_dems, **params, **builder_kwargs
        )
        y_train = torch.tensor([sample[1][0] for sample in train_set])
        class_weights = compute_balanced_class_weights(y_train)

        val_auroc = train_and_eval(
            model, *data, lr=lr, class_weights=class_weights, trial=trial, epochs=epochs
        )

        return val_auroc

    return objective


def train_and_eval(
    model,
    train_loader,
    early_stopping_loader,
    val_loader,
    lr,
    class_weights,
    trial,
    epochs=100,
):
    criterion, optimizer, scheduler, early_stopper = set_up_training_components(
        model, lr, class_weights
    )
    n_epochs = 100

    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        # Train
        epoch_train_loss = run_1_epoch(
            model, train_loader, criterion, optimizer, train=True
        )
        # Evaluate on early stopping set and report intermediate balanced_accuracy
        epoch_es_loss, epoch_es_auroc = run_1_epoch(
            model,
            early_stopping_loader,
            criterion,
            train=False,
            additional_metric=roc_auc_score,
        )
        scheduler.step(epoch_es_loss)
        trial.report(epoch_es_auroc, epoch)

        if early_stopper.early_stop(epoch_es_loss):
            print(f"Early stopping triggered. Stopping at epoch {epoch+1}")
            break
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        gc.collect()
        tqdm.write(
            f"Epoch {epoch+1}/{n_epochs} |"
            f"Train loss: {epoch_train_loss:.4f} | Early stopping loss: {epoch_es_loss:.4f} |"
            f" Learning Rate: {scheduler.get_last_lr()[0]}"
        )

    val_loss, val_auroc = run_1_epoch(
        model, val_loader, criterion, train=False, additional_metric=roc_auc_score
    )
    return val_auroc


###### Define search spaces ###########
def lstm_params(trial):
    return {
        "lr": 0.0001,  # trial.suggest_float('lr', 1e-6, 1e-1, log=True)
        "nhidden": trial.suggest_int("nhidden", 16, 64),
        "n_layers": trial.suggest_int("n_layers", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.1, 0.9),
    }


def gru_params(trial):
    return {
        "lr": 0.0001,  # trial.suggest_float('lr', 1e-6, 1e-1, log=True)
        "nhidden": trial.suggest_int("nhidden", 16, 64),
        "n_layers": trial.suggest_int("n_layers", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.1, 0.9),
    }


def grud_params(trial):
    return {
        "lr": 0.0001,  # trial.suggest_float('lr', 1e-6, 1e-1, log=True)
        "nhidden": trial.suggest_int("nhidden", 8, 64), # 16 for min?
        "dropout": trial.suggest_float("dropout", 0.1, 0.9),
    }


def lr_params(trial):
    penalty_solver_pair = trial.suggest_categorical(
        "penalty_solver_pair",
        [
            "l1,liblinear",
            "l1,saga",
            "l2,lbfgs",
            "l2,liblinear",
            "l2,sag",
            "l2,saga",
            "elasticnet,saga",
        ],
    )
    l1_ratio = trial.suggest_float("l1_ratio", 1e-4, 0.99)
    if penalty_solver_pair.split(",")[0] != "elasticnet":
        l1_ratio = None
    C = trial.suggest_float("C", 0, 1)
    return {
        "penalty": penalty_solver_pair.split(",")[0],
        "solver": penalty_solver_pair.split(",")[1],
        "l1_ratio": l1_ratio,
        "C": C,
    }


def xgb_params(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_split_loss": trial.suggest_float("min_split_loss", 0.1, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.00001, 0.1, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 1000),
        "n_estimators": trial.suggest_int("n_estimators", 4, 50),
    }


def nn_params(trial):
    hidden_layer_sizes = []
    n_layers = trial.suggest_int("n_layers", 1, 3)
    for n in range(n_layers):
        hidden_layer_sizes += [
            trial.suggest_int(f"N_neurons_in_{n}_layer", 5, 50, step=5)
        ]
    activation = trial.suggest_categorical("activation", ["logistic", "tanh", "relu"])
    solver = trial.suggest_categorical("solver", ["lbfgs", "adam"])
    alpha = trial.suggest_float("alpha", 0.1, 1000, log=True)
    learning_rate = trial.suggest_categorical(
        "learning_rate", ["constant", "invscaling", "adaptive"]
    )
    tol = trial.suggest_float("tol", 0.0001, 0.2)
    return {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "solver": solver,
        "alpha": alpha,
        "learning_rate": learning_rate,
        "tol": tol,
    }


def build_nn(params):
    return MLPClassifier(**params, early_stopping=True, n_iter_no_change=5)


def build_xgb(params):
    return xgb.XGBClassifier(**params)


def build_lr(**params):
    return LogisticRegression(
        penalty=params["penalty"],
        solver=params["solver"],
        l1_ratio=params["l1_ratio"],
        C=params["C"],
    )


def build_lstm(params, n_features, use_dems, **builder_kwargs):
    return dl_models.Hiddenstate_LSTM(
        n_features=n_features,
        nhidden=params["nhidden"],
        n_layers=params["n_layers"],
        dropout=params["dropout"],
        use_static=use_dems,
    )


def build_grud(**builder_kwargs):
    X_mean = builder_kwargs["X_mean"]
    n_features = builder_kwargs["n_features"]
    use_static = builder_kwargs["use_static"]
    dropout = builder_kwargs["dropout"]
    nhidden = builder_kwargs["nhidden"]
    return dl_models.GRUD(
        input_size=n_features,
        nhidden=nhidden,
        use_static=use_static,
        X_mean=X_mean,
        dropout=dropout,
    )


MODEL_REGISTRY = {
    "LSTM_ALT": {
        "param_func": lstm_params,
        "builder": model_dispatcher["LSTM_ALT"],  # build_lstm,
        "collate_fn": custom_collate_function,
    },
    "GRUD": {
        "param_func": grud_params,
        "builder": model_dispatcher["GRUD"],
        "collate_fn": custom_collate_function_grud,
    },
    "GRU": {
        "param_func": gru_params,
        "builder": model_dispatcher["GRU"],
        "collate_fn": custom_collate_function,
    },
    "LR": {
        "param_func": lr_params,
        "builder": model_dispatcher["LR"],
    },
    "XGB": {"param_func": xgb_params, "builder": model_dispatcher["XGB"]},
    "NN": {"param_func": nn_params, "builder": model_dispatcher["NN"]}
}
