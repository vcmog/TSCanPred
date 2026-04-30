import os
import argparse
import gc
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from datasets.preparation.grud_prep import generate_GRUD_array, generate_delta_mask_sequence
from datasets.collate import custom_collate_function_grud
from datasets.sequence_datasets import compile_grud_dataset
from datasets.utils.utils import get_loader_with_batch_sampling, apply_transforms
from evaluation.evaluation import evaluate_performance_GRUD
from models.dl_models import GRUD
from utils.data import compute_balanced_class_weights
from utils.io_utils import ensure_dir_exists
from datasets.preparation.common import get_dem_variables
from utils.splits import get_kfold_split_indices, get_datasplits
from utils.results import build_results_structures
from training.torch.training import train_model
from hyperparameter_tune.hyperparameter_dispatcher import get_hyperparameters
from hyperparameter_tune.hyperparameter_tune import tune_torch_model_from_datasets

def cross_val_grud(event_df,demographic_data, min_seq_len, max_seq_len=25, hyperparameter_tune=True):

    data, all_outcomes, patient_ids = generate_GRUD_array(event_df, max_seq_len)
    demographic_data = demographic_data.loc[patient_ids].values
    BATCH_SIZE = 48
    LR = 0.001  # 0.0005
    
    n_patients = len(all_outcomes) # note: this isn't the correct number of patients: excluding short sequences happens in generate_delta_mask
    n_cases = sum(all_outcomes)
    n_controls = n_patients - n_cases
    sum_stats = pd.DataFrame(
        {"N_patients": n_patients, "N_cases": n_cases, "N_controls": n_controls},
        index=[min_seq_len],
    )
    sum_stats.index.name = "min_seq_len"

    splits = get_kfold_split_indices(patient_ids, all_outcomes)
    (
        train_results,
        test_results,
        train_predictions,
        test_predictions,
    ) = build_results_structures()

    
    best_fold_auc = 0
    for fold, (dev_ids, test_ids) in enumerate(splits):

        (
            dev_data,
            test_data,
            dev_outcomes,
            test_outcomes,
            dev_ids,
            test_ids,
            dev_dems,
            test_dems,
        ) = get_datasplits(
            data,
            all_outcomes,
            dev_ids,
            test_ids,
            patient_ids=patient_ids,
            all_dems=demographic_data,
        )

        (
            dataset_combined,
            dev_lengths,
            X_mean,
            train_means,
            train_std,
            max_delta,
        ) = generate_delta_mask_sequence(dev_data, return_descriptors=True)
        print("Masks made for train")
        test_data, test_lengths = generate_delta_mask_sequence(
            test_data, train_means=train_means, train_stds=train_std, max_delta=max_delta
        )
        print("Masks made for test")

        dev_set = compile_grud_dataset(
            dataset_combined,
            dev_outcomes,
            dev_lengths,
            dev_ids,
            dems=dev_dems,
            length_required=min_seq_len,
            max_length=max_seq_len,
        )

        test_set = compile_grud_dataset(
            test_data,
            test_outcomes,
            test_lengths,
            test_ids,
            test_dems,
            length_required=min_seq_len,
            max_length=max_seq_len,
        )

        dev_set, test_set = apply_transforms(
            train_set=dev_set, test_set=test_set, variable_dim=2, model_type="GRUD"
        )

        train_set, val_set = torch.utils.data.random_split(dev_set, [0.8, 0.2])
        if hyperparameter_tune is True:
            # tune_GRUD_from_datasets(train_set, val_set, 250, X_mean=X_mean)
            tune_torch_model_from_datasets(
                "GRUD",
                train_set,
                val_set,
                n_trials=100,
                use_dems=True,
                n_features=X_mean.shape[-1],
                X_mean=X_mean,
            )
            ### ADD IN TO SAVE AND LOAD THESE
        best_params = get_hyperparameters(
            "GRUD"
        )  # best_params ={'nhidden':33, 'dropout':0.25506139548130563}

        train_loader = get_loader_with_batch_sampling(
            train_set,
            batch_size=BATCH_SIZE,
            num_bins=15,
            collate_fn=custom_collate_function_grud,
        )
        val_loader = get_loader_with_batch_sampling(
            val_set,
            batch_size=BATCH_SIZE,
            num_bins=15,
            collate_fn=custom_collate_function_grud,
        )
        test_loader = DataLoader(test_set, shuffle=False, batch_size=1)

        dev_set_outcomes = dev_set.labels
        class_weights = compute_balanced_class_weights(dev_set_outcomes)

        n_features = X_mean.shape[-1]
        print("N_features:", n_features)
        model = GRUD(input_size=34, X_mean=X_mean, use_static=True, **best_params)

        train_loss, val_loss, _ = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=100,
            lr=LR,
            pos_class_weight=class_weights[0],
            patience=5,
        )

        fold_train_results, fold_train_predictions = evaluate_performance_GRUD(
            model, train_loader, return_predictions=True
        )

        fold_test_results, fold_test_predictions = evaluate_performance_GRUD(
            model, test_loader, return_predictions=True
        )

        fold_train_results["fold"] = fold_train_predictions["fold"] = fold_test_results[
            "fold"
        ] = fold_test_predictions["fold"] = fold
        if fold_train_results["auroc"] > best_fold_auc:
            best_overall_params = best_params
            best_fold_auc = fold_train_results["auroc"]
        print("Model evaluated.")
        print(f"Model performance on the test set for fold {fold}:  ", test_results)
        train_results.loc[fold] = fold_train_results
        test_results.loc[fold] = fold_test_results
        train_predictions.append(fold_train_predictions)
        test_predictions.append(fold_test_predictions)
        print("Model evaluated.")
        print("Model performance on the train set:  ", train_results)
        print("Model performance on the test set:  ", test_results)

    


    train_predictions = pd.concat(train_predictions)
    test_predictions = pd.concat(test_predictions)

    print(f"Done. Model GRUD trained and evaluated at lead_time={lead_time} days.")
    
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


print("Reading in .csv files...")
blood_data = pd.read_csv(
        input_data_dir + r"\cohort_bloods.csv",
        index_col=0,
        dtype=dtypes,
        parse_dates=date_cols,
    )
demographic_data = pd.read_csv(demographicpath, index_col=0)


total_case = blood_data[blood_data["OUTCOME"] == 1]["PATIENT_ID"].nunique()
total_control = blood_data[blood_data["OUTCOME"] == 0]["PATIENT_ID"].nunique()

min_seq_lens = [8]

dir = rf"S:\1475 - Early detection of cancer occurrence and recurrence in primary care using ar\Data\{cancer_site} Cancer Prediction\SeqLenEffectGRUD"
ensure_dir_exists(dir)
dir += fr"\lead_time={lead_time}"
ensure_dir_exists(dir)
for min_seq_len in min_seq_lens:

    results, sum_stats = cross_val_grud(blood_data, demographic_data, min_seq_len)

    results.to_csv(
        rf"S:\1475 - Early detection of cancer occurrence and recurrence in primary care using ar\Data\{cancer_site} Cancer Prediction\SeqLenEffectGRUD\lead_time={lead_time}\minseqlen={min_seq_len}_results"
    )
    sum_stats.to_csv(
        rf"S:\1475 - Early detection of cancer occurrence and recurrence in primary care using ar\Data\{cancer_site} Cancer Prediction\SeqLenEffectGRUD\lead_time={lead_time}\minseqlen={min_seq_len}_n_patients"
    )
    del results, sum_stats
    gc.collect()