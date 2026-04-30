import argparse
import time
import os
import json
import pandas as pd
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from torch.utils.data import DataLoader
import config
from hyperparameter_tune.hyperparameter_tune import tune_torch_model_from_datasets
from hyperparameter_tune.hyperparameter_dispatcher import get_hyperparameters
from datasets.sequence_datasets import compile_grud_dataset
from datasets.preparation.grud_prep import (
    generate_GRUD_array,
    generate_delta_mask_sequence,
)
from datasets.preparation.common import get_dem_variables
from datasets.utils.utils import get_loader_with_batch_sampling, apply_transforms
from datasets.collate import custom_collate_function_grud
from training.torch.training import train_model
from training.sklearn.cross_validation import get_kfold_split_indices, get_datasplits
from evaluation.evaluation import evaluate_performance_GRUD
from models.dl_models import GRUD
from utils.data import compute_balanced_class_weights
from utils.io_utils import save_training_curve, save_fold_results
import utils.io_utils as gu
from utils.results import build_results_structures, save_results_dfs, update_with_previous_fold_results


if __name__ == "__main__":
    # initialise argument parser
    parser = argparse.ArgumentParser()

    # add arguments to parse
    parser.add_argument("--lead_time", type=int, default=None, required=True)
    parser.add_argument("--hyperparameter_tune", action="store_true")
    parser.add_argument("--earlystop_patience", type=int, default=5, required=False)
    parser.add_argument("--earlystop_delta", type=float, default=0.01, required=False)
    parser.add_argument("--use_static", action="store_true", default=False)
    parser.add_argument("--min_seq_len", type=int, required=False, default=2)
    parser.add_argument("--max_seq_len", type=int, required=False, default=20)
    parser.add_argument("--config", type=str, default=None, required=True)
    parser.add_argument("--start_fold", type=int, default=0)
    parser.set_defaults(hyperparameter_tune=False)
    args = parser.parse_args()

lead_time = args.lead_time
hyperparameter_tune = args.hyperparameter_tune
earlystop_patience = args.earlystop_patience
earlystop_delta = args.earlystop_delta
use_static = args.use_static
min_seq_len = args.min_seq_len
max_seq_len = args.max_seq_len
config_file = args.config
START_FOLD = args.start_fold

if config_file is not None:
    project_dir, input_dir, output_dir, model_dir = config.get_dirs(config_file)

experiment_input_dir = input_dir + rf"\lead_time={lead_time}"

input_data_dir = experiment_input_dir
if not os.path.exists(experiment_input_dir):
    os.mkdir(experiment_input_dir)

start_time = time.time()

if use_static is True:
    STATIC_STR = "_with_static"
else:
    STATIC_STR = ""


model_dir, output_dir = gu.make_crossval_directories(
    lead_time, STATIC_STR, model_name="GRUD", config_file=config_file
)

dtypes = {
    "ALF_PE": "Int64",
    "OUTCOME": "Int16",
    "PATIENT_ID": "Int64",
    "TEST_LABEL": str,
    "EVENT_VAL": "float64",
}
date_cols = ["OBSERVATION_START", "PSEUDO_INDEX", "EVENT_DT"]


print("Reading in .csv files...")
event_df = pd.read_csv(
    input_data_dir + r"\cohort_bloods.csv",
    index_col=0,
    dtype=dtypes,
    parse_dates=date_cols,
)


data, all_outcomes, patient_ids = generate_GRUD_array(event_df, 25)
all_dems = get_dem_variables(input_data_dir, patient_ids)
BATCH_SIZE = 48
LR = 0.001  # 0.0005


splits = get_kfold_split_indices(patient_ids, all_outcomes)
(
    train_results,
    test_results,
    train_predictions,
    test_predictions,
) = build_results_structures()

update_with_previous_fold_results(
    model_name="GRUD", 
                                  start_fold=START_FOLD, 
                                  output_dir=output_dir,
                                  static_str=STATIC_STR,
                                  train_results=train_results,
                                  test_results=test_results,
                                  train_predictions=train_predictions,
                                  test_predictions=test_predictions
)
best_overall_params = {}
best_fold_auc = 0
for fold, (dev_ids, test_ids) in enumerate(splits):
    if fold < START_FOLD:
        continue
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
        all_dems=all_dems,
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
            use_dems=use_static,
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
    save_fold_results(
        model_name="GRUD",
        output_dir=output_dir,
        static_str=STATIC_STR,
        fold=fold,
        fold_train_results=fold_train_results,
        fold_test_results=fold_test_results,
        fold_train_predictions=fold_train_predictions,
        fold_test_predictions=fold_test_predictions,
    )
    print("Model evaluated.")
    print("Model performance on the train set:  ", train_results)
    print("Model performance on the test set:  ", test_results)
    save_training_curve(
        train_loss,
        val_loss,
        output_dir + rf"\GRUD\cross-val{STATIC_STR}\{fold}_training_curve.pdf",
    )

file_name = model_dir + r"\optimal_hyperparameters_GRU_D"
with open(file_name, "w") as f:
    json.dump(best_overall_params, f)


train_predictions = pd.concat(train_predictions)
test_predictions = pd.concat(test_predictions)
save_results_dfs(
    model_name="GRUD",
    train_results=train_results,
    test_results=test_results,
    train_predictions=train_predictions,
    test_predictions=test_predictions,
    static_str=STATIC_STR,
    output_dir=output_dir,
)

print(f"Done. Model GRUD trained and evaluated at lead_time={lead_time} days.")
print(f"Results saved to: {experiment_input_dir}")

end_time = time.time()

elapsed_time = (end_time - start_time) / 60
print(f"Script took {elapsed_time:.2f} minutes to run.")
