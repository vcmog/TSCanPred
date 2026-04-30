import pandas as pd
import numpy as np
from datasets.utils.utils import remove_short_sequences_pandas, get_dem_variables
from datasets.preparation import rnn_prep, cnn_prep
from utils.data import compute_balanced_class_weights
from datasets.utils.time_utils import get_time_diff_for_consecutive_tests
from datasets.utils.utils import (
    dataset_split,
    apply_transforms,
    get_loader_with_batch_sampling,
)
from torch.utils.data import DataLoader


def prepare_numpy_files_with_split(
    input_dir, experiment_input_dir, model, lead_time, max_length, pad=True
):
    """ "

    Args:
        input_dir (str): Where the input csvs or numpy files are stored.
        model (str): "RNN" or "CNN".
        existing_numpy_file (bool): Whether the .npy files are stored already and should be used.
    Returns:
        n_features (int):   How many variables are used.
        max_len (int):      How many timesteps there are for each patient.
        class_weights:      Class_weights for the proportion of patients.
    """

    dtypes = {
        "ALF_PE": "Int64",
        "OUTCOME": "Int16",
        "PATIENT_ID": "Int64",
        "TEST_LABEL": str,
        "EVENT_VAL": "float64",
    }
    date_cols = ["OBSERVATION_START", "PSEUDO_INDEX", "EVENT_DT"]

    print("Reading in .csv files...")
    train_set = pd.read_csv(
        input_dir + r"\train.csv", index_col=0, dtype=dtypes, parse_dates=date_cols
    )

    test_set = pd.read_csv(
        input_dir + r"\test.csv",
        index_col=0,
        dtype=dtypes,
        parse_dates=date_cols,
    )
    print("Data loaded from CSV.")

    print("Creating numpy arrays...")
    if model == "RNN-sequenceclass":
        X_train, y_train, train_ids = rnn_prep.generate_RNN_input(
            train_set, lead_time=lead_time, max_length=max_length, pad=True
        )
        max_len = X_train.shape[1]
        X_test, y_test, test_ids = rnn_prep.generate_RNN_input(
            test_set, max_length=max_len, lead_time=lead_time, pad=True
        )
        n_features = X_train.shape[2]
        print("Number of features:  ", n_features)
        print("Max sequence length: ", max_len)
    elif model == "RNN-lasthiddenstate":
        X_train, y_train, train_ids = rnn_prep.generate_RNN_input(
            train_set,
            lead_time=lead_time,
            zero_fill_nan=False,
            max_length=None,
            pad=False,
            return_lengths=False,
        )
        lengths = [len(x) for x in X_train[0]]
        max_len = max(lengths)  # X[0][0].shape[0]
        n_features = X_train[0][0].shape[0]  # X[0][0].shape[1]
        X_test, y_test, test_ids = rnn_prep.generate_RNN_input(
            test_set,
            lead_time=lead_time,
            zero_fill_nan=False,
            max_length=None,
            pad=False,
            return_lengths=False,
        )
    elif model == "CNN":
        X_train, y_train, train_ids = cnn_prep.generate_CNN_input(
            train_set, lead_time=lead_time, max_len=max_length
        )
        max_len = X_train.shape[2]
        X_test, y_test, test_ids = cnn_prep.generate_CNN_input(
            test_set, max_len=max_len, lead_time=lead_time
        )
        n_features = X_train.shape[1]
        print("Number of features:  ", n_features)
        print("Max sequence length: ", max_len)
    print("Numpy arrays created.")

    class_weights = compute_balanced_class_weights(y_train)
    print("Saving arrays...")
    train_dems, test_dems = get_dem_variables(experiment_input_dir, train_ids, test_ids)
    save_arrays(
        experiment_input_dir,
        model,
        X_train=X_train,
        y_train=y_train,
        train_ids=train_ids,
        test_ids=test_ids,
        X_test=X_test,
        y_test=y_test,
        train_dems=train_dems,
        test_dems=test_dems,
    )
    return n_features, max_len, class_weights


# def save_arrays(experiment_input_dir, model, X_train, y_train, train_ids,test_ids, X_test, y_test, train_dems, test_dems):
#
#    np.save(experiment_input_dir+fr'\{model}_test_inputs.npy', X_test)
#    np.save(experiment_input_dir+fr'\{model}_test_labels.npy', y_test.flatten())
#    np.save(experiment_input_dir+fr'\{model}_test_ids.npy', test_ids.flatten())
#    np.save(experiment_input_dir+fr'\{model}_test_dems.npy', test_dems)
#    print('Numpy arrays saved.')


def save_arrays(
    experiment_input_dir,
    model,
    X_train,
    y_train,
    train_ids,
    train_dems,
    X_test=None,
    y_test=None,
    test_ids=None,
    test_dems=None,
):
    if X_test:
        prefix_str = f"\{model}_test_"
        np.save(experiment_input_dir + rf"{prefix_str}_inputs.npy", X_test)
        np.save(experiment_input_dir + rf"{prefix_str}_labels.npy", y_test.flatten())
        np.save(experiment_input_dir + rf"{prefix_str}_ids.npy", test_ids.flatten())
        np.save(experiment_input_dir + rf"{prefix_str}_dems.npy", test_dems)
        prefix_str = f"\{model}_train_"
    else:
        prefix_str = f"\{model}"
    np.save(experiment_input_dir + rf"{prefix_str}_inputs.npy", X_train)
    np.save(experiment_input_dir + rf"{prefix_str}_labels.npy", y_train.flatten())
    np.save(experiment_input_dir + rf"{prefix_str}_ids.npy", train_ids.flatten())
    np.save(experiment_input_dir + rf"{prefix_str}_dems.npy", train_dems)
    print("Numpy arrays saved.")


def prepare_numpy_files_nosplit(
    input_dir,
    experiment_input_dir,
    model,
    lead_time,
    max_length,
    min_length=None,
    return_lengths=False,
):
    """ "

    Args:
        input_dir (str): Where the input csvs or numpy files are stored.
        model (str): "RNN" or "CNN".
        existing_numpy_file (bool): Whether the .npy files are stored already and should be used.
    Returns:
        n_features (int):   How many variables are used.
        max_len (int):      How many timesteps there are for each patient.
        class_weights:      Class_weights for the proportion of patients.
    """

    dtypes = {
        "ALF_PE": "Int64",
        "OUTCOME": "Int16",
        "PATIENT_ID": "Int64",
        "TEST_LABEL": str,
        "EVENT_VAL": "float64",
    }
    date_cols = ["OBSERVATION_START", "PSEUDO_INDEX", "EVENT_DT"]

    print("Reading in .csv files...")
    dataset = pd.read_csv(
        input_dir + r"\cohort_bloods.csv",
        index_col=0,
        dtype=dtypes,
        parse_dates=date_cols,
    )

    print("Data loaded from CSV.")

    print("Creating numpy arrays...")
    if model == "RNN-sequenceclass":
        X, y, ids = rnn_prep.generate_RNN_input(
            dataset,
            lead_time=lead_time,
            zero_fill_nan=False,
            max_length=max_length,
            pad=True,
            return_lengths=False,
            min_length=min_length,
        )
        max_len = X.shape[1]
        n_features = X.shape[2]
        print("Number of features:  ", n_features)
        print("Max sequence length: ", max_len)
    elif model == "RNN-lasthiddenstate":
        X, y, ids = rnn_prep.generate_RNN_input(
            dataset,
            lead_time=lead_time,
            zero_fill_nan=False,
            max_length=None,
            pad=False,
            return_lengths=False,
            min_length=min_length,
        )
        lengths = [len(x) for x in X[0]]
        max_len = max(lengths)
        n_features = X[0][0].shape[0]

    elif model == "CNN":
        X, y, ids = rnn_prep.generate_CNN_input(
            dataset, lead_time=lead_time, max_len=max_length
        )

        max_len = X.shape[2]
        n_features = X.shape[1]
        print("Number of features:  ", n_features)
        print("Max sequence length: ", max_len)
    print("Numpy arrays created.")

    class_weights = compute_balanced_class_weights(y)
    print("Saving arrays...")
    dems = get_dem_variables(experiment_input_dir, ids)
    # np.save(experiment_input_dir+fr'\{model}_seq_lengths.npy', lengths)
    save_arrays(experiment_input_dir, model, X, y, ids, dems)
    return n_features, max_len, class_weights


def initialise_dataloaders(
    train_set,
    test_set,
    batch_size,
    variable_dim,
    shuffle=True,
    num_workers=0,
    split_train=True,
    train_val_split=0.8,
    collate_fn=None,
    return_datasets=False,
    val_set=None,
):
    """
    Initialise dataloaders and normalise the dataset.
    Args:


        variable_dim (int): Which dimension the variables are in.
             E.g., for input with dimensions (batch_size, n_variables, seq_len) variable_dim would be 1.

    """
    # Split the train set indices into training and validation sets
    if split_train:
        train_set, val_set = dataset_split(train_set, proportion=train_val_split)

    train_set, test_set, val_set = apply_transforms(
        train_set=train_set, test_set=test_set, variable_dim=variable_dim, val_set=val_set
    )

    train_loader = get_loader_with_batch_sampling(
        train_set, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    # get_loader_with_batch_sampling(test_set, batch_size=8, collate_fn=collate_fn, num_workers=num_workers)
    return_items = [train_loader, test_loader]
    if val_set:
        val_loader = get_loader_with_batch_sampling(
            val_set,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        return_items.insert(1, val_loader)
    if return_datasets == True:
        return_items.append((train_set, val_set, test_set))
    return return_items
