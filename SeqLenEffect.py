import argparse
import config
from models import dl_models, model_dispatcher
from hyperparameter_tune import hyperparameter_dispatcher
from torch.utils.data import Subset, Dataset
from pathlib import Path
from evaluation.evaluation import evaluate_performance_torchmodel
from utils.data import count_outcomes_from_dataloader, compute_balanced_class_weights
from datasets.utils.utils import dataset_split
from utils.io_utils import ensure_dir_exists
from datasets.preparation.common import initialise_dataloaders, apply_transforms
from datasets.collate import custom_collate_function
from training.torch.training import train_model
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from hyperparameter_tune.hyperparameter_tune import tune_torch_model_from_datasets
from datasets.preparation.rnn_prep import generate_RNN_input
import plotly.express as px



class Local_Dataset(Dataset):
    def __init__(self, data, labels, ids, dems):
        self.data = data
        if isinstance(self.data, object) and len(self.data[0]) == 2:
            self.data, self.lengths = zip(*self.data)
            self.data = np.array(self.data, dtype=float)
            self.lengths = np.array(self.lengths, dtype=float)
        else:
            self.lengths = None

        self.labels = labels
        self.ids = ids
        self.static_data = dems
        self.ages = [age for age, _ in self.static_data]
        self.dtype = torch.float32
        #self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        pat_id = self.ids[idx]
        label = torch.tensor([label], dtype=self.dtype)
        sample = torch.tensor(np.array(sample), dtype=self.dtype)
        static_data = self.static_data[idx]
        static_data = torch.tensor(static_data, dtype=self.dtype)
        #if self.transform:
        #    sample = self.transform(sample)
        pat_id = torch.tensor([pat_id], dtype=self.dtype)
        return_vals = [sample, label, pat_id, static_data]
        if self.lengths:
            length = self.lengths[idx]
            length = torch.tensor(length, dtype=self.dtype)
            return_vals.append(length)

        return tuple(return_vals)


def cross_val_model(blood_data, demographic_data, min_seq_len, model_name):

    patient_data, labels, patient_ids = generate_RNN_input(
        blood_data,
        lead_time=0,
        zero_fill_nan=False,
        max_length=max_seq_len,
        pad=False,
        return_lengths=False,
        min_length=min_seq_len,
    )
    dems = demographic_data.loc[patient_ids].values
    n_patients = len(labels)
    n_cases = sum(labels)
    n_controls = n_patients - n_cases
    
    sum_stats = pd.DataFrame(
        {"N_patients": n_patients, "N_cases": n_cases, "N_controls": n_controls},
        index=[min_seq_len],
    )
    sum_stats.index.name = "min_seq_len"

    dataset = Local_Dataset(patient_data, labels, patient_ids, dems)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=36)

    train_results = pd.DataFrame(
        columns=[
            "accuracy",
            "precision",
            "recall",
            "conf_matrix",
            "auroc",
            "average_precision",
            "balanced_accuracy",
            "fold",
        ]
    )
    test_results = train_results.copy()

    train_predictions = []

    test_predictions = []

    data_set_labels = [i[1] for i in dataset]
    k_folds = kf.split(dataset, data_set_labels)

    for fold, (fold_train, fold_test) in enumerate(k_folds):

        train_set = Subset(dataset, fold_train)
        test_set = Subset(dataset, fold_test)

        # this matches what happens in initialise datalaoders but means I can use the val set for hyperparameter
        # tuning without bothering with the dataloaders yet:
        train_set, val_set = dataset_split(train_set, 0.8)
        train_set, test_set, val_set = apply_transforms(
        train_set=train_set, test_set=test_set, variable_dim=2, val_set=val_set
    )
        first_sample_data = train_set[0][0]
        n_features = first_sample_data.size(1)
        # train_set, val_set, test_set = datasets
        #tune_lstm_alt_from_datasets(train_set, val_set, n_trials=100, use_dems=True)
        tune_torch_model_from_datasets(model_name,dev_set=train_set, val_set=val_set, n_trials=100, use_dems=True, n_features=n_features)
        best_params = hyperparameter_dispatcher.get_hyperparameters(
            model_name=model_name
        )

        lr = 0.0001  # best_params.pop('lr')
        batch_size = 32  # 2**best_params.pop('n_power')

        train_loader, val_loader, test_loader, datasets = initialise_dataloaders(
            train_set,
            test_set,
            batch_size=batch_size,
            variable_dim=2,
            split_train=False,
            collate_fn=custom_collate_function,
            return_datasets=True,
            val_set=val_set,
        )

        train_outcomes, train_len = count_outcomes_from_dataloader(
            train_loader
        )
        val_outcomes, val_len = count_outcomes_from_dataloader(val_loader)
        test_outcomes, test_len = count_outcomes_from_dataloader(test_loader)

        print("Train set:", train_len, train_outcomes, train_outcomes / train_len)
        print("Val set:", val_len, val_outcomes, val_outcomes / val_len)
        print("Test set:", test_len, test_outcomes, test_outcomes / test_len)

        first_patient = train_set[0]
        first_patient_inputs = first_patient[0]
        n_features = first_patient_inputs.size(1)

        y_train = dataset.labels[fold_train]
        class_weights = compute_balanced_class_weights(y_train)

        model = model_dispatcher.model_dispatcher[model_name](n_features=n_features, use_static=True, **best_params)
        #model = dl_models.Hiddenstate_LSTM(n_features, use_static=True, **best_params)

        train_losses, val_losses, best_val_loss = train_model(
            model,
            train_loader,
            val_loader,
            lr=lr,
            num_epochs=n_epochs,
            pos_class_weight=class_weights[0],
            patience=earlystop_patience,
            min_earlystop_delta=earlystop_delta,
        )

        fig = px.line(
            y=[train_losses, val_losses],
            title=f"Training curve for fold {fold}",
            labels=["Training", " Validation"],
        )
        fig.show()
        fold_train_results, fold_train_predictions = evaluate_performance_torchmodel(
            model, train_loader, return_predictions=True
        )

        fold_test_results, fold_test_predictions = evaluate_performance_torchmodel(
            model, test_loader, return_predictions=True
        )

        fold_train_results["fold"] = fold_train_predictions["fold"] = fold_test_results[
            "fold"
        ] = fold_test_predictions["fold"] = fold

        train_results.loc[fold] = fold_train_results
        test_results.loc[fold] = fold_test_results
        train_predictions.append(fold_train_predictions)
        test_predictions.append(fold_test_predictions)

        print("Model evaluated.")
        print(f"Model performance on the test set for fold {fold}:  ", test_results)

    print("Model evaluated.")
    print("Model performance on the train set:  ", train_results)
    print("Model performance on the test set:  ", test_results)

    return test_results, sum_stats


if __name__ == "__main__":
    # initialise argument parser
    parser = argparse.ArgumentParser()

    # add arguments to parse

    parser.add_argument("--n_epochs", type=int, default=100, required=False)
    parser.add_argument("--use_dems", dest="use_dems", action="store_true")
    parser.add_argument("--earlystop_patience", type=int, default=5, required=False)
    parser.add_argument("--earlystop_delta", type=float, required=False, default=0.1)
    parser.add_argument("--max_seq_len", type=int, required=False, default=None)
    parser.add_argument("--lead_time", type=int, required=True)
    parser.add_argument("--cancer_site", required=True, choices=["OG", "Panc"])
    parser.add_argument("--model_name", required=True, choices=['LSTM_ALT','GRU'])
    # parser.add_argument("--min_seq_lens", type=list, required=True, default=None)
    parser.set_defaults(use_dems=False)

    args = parser.parse_args()

n_epochs = args.n_epochs
use_dems = args.use_dems
earlystop_patience = args.earlystop_patience
earlystop_delta = args.earlystop_delta
max_seq_len = args.max_seq_len
lead_time = args.lead_time
cancer_site = args.cancer_site
model_name = args.model_name

input_data_dir = rf"S:\1475 - Early detection of cancer occurrence and recurrence in primary care using ar\Data\{cancer_site} Cancer Prediction\input\lead_time={lead_time}"
blood_path = input_data_dir + r"\cohort_bloods.csv"
demographicpath = input_data_dir + r"\demographics.csv"

dtypes = {
    "ALF_PE": "Int64",
    "OUTCOME": "Int16",
    "PATIENT_ID": "Int64",
    "TEST_LABEL": str,
    "EVENT_VAL": "float64",
}
date_cols = ["OBSERVATION_START", "PSEUDO_INDEX", "EVENT_DT"]
blood_data = pd.read_csv(blood_path, index_col=0, dtype=dtypes, parse_dates=date_cols)
demographic_data = pd.read_csv(demographicpath, index_col=0)

total_case = blood_data[blood_data["OUTCOME"] == 1]["PATIENT_ID"].nunique()
total_control = blood_data[blood_data["OUTCOME"] == 0]["PATIENT_ID"].nunique()

min_seq_lens = [1,2,3, 4, 5, 6, 8, 10] #1,2,3, 4, 5, 6, 8,10
dir = rf"S:\1475 - Early detection of cancer occurrence and recurrence in primary care using ar\Data\{cancer_site} Cancer Prediction\SeqLenEffect"
ensure_dir_exists(dir)
dir += fr"\lead_time={lead_time}"
ensure_dir_exists(dir)
for min_seq_len in min_seq_lens:

    results, sum_stats = cross_val_model(blood_data, demographic_data=demographic_data, min_seq_len = min_seq_len, model_name=model_name)

    results.to_csv(
        rf"S:\1475 - Early detection of cancer occurrence and recurrence in primary care using ar\Data\{cancer_site} Cancer Prediction\SeqLenEffect\lead_time={lead_time}\{model_name}_minseqlen={min_seq_len}_results"
    )
    sum_stats.to_csv(
        rf"S:\1475 - Early detection of cancer occurrence and recurrence in primary care using ar\Data\{cancer_site} Cancer Prediction\SeqLenEffect\lead_time={lead_time}\{model_name}_minseqlen={min_seq_len}_n_patients"
    )
plt.show()
