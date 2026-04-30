import pandas as pd
from collections import Counter
import numpy as np

def build_results_structures(get_importances=False, get_feature_use_counter=False):
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

    feature_use_counter = Counter()

    importances = {}

    return_items = [train_results, test_results, train_predictions, test_predictions]

    if get_feature_use_counter is True:
        return_items.append(feature_use_counter)

    if get_importances is True:
        return_items.append(importances)
    return return_items


def save_results_dfs(
    model_name,
    train_results,
    test_results,
    train_predictions,
    test_predictions,
    static_str,
    output_dir,
    feature_use_counter=None,
    importances=None,
    feature_names=None,
    save_dir=None,
    feature_set_name=""
):
    if save_dir is None:
        save_dir = output_dir + rf"\{model_name}\cross-val{static_str}{feature_set_name}"
    train_results.to_csv(
        save_dir + r"\results\train.csv"
    )
    test_results.to_csv(
        save_dir + rf"\results\test.csv"
    )

    train_predictions.to_csv(
        save_dir+ rf"\\predictions\train.csv"
    )
    test_predictions.to_csv(
        save_dir + rf"\predictions\test.csv"
    )

    if feature_use_counter is not None:
        feature_use_counter.to_csv(
            save_dir + rf"\feature_use_counter.csv"
        )

    if importances is not None:
        importances.to_csv(
            save_dir + rf"\feature_importances.csv"
        )

    if feature_names is not None:
        np.savetxt(
            save_dir + rf"\feature_names.csv",
            feature_names,
            delimiter=",",
            fmt='%s'
        )


def update_with_previous_fold_results(model_name, start_fold, output_dir, static_str, train_results, test_results, train_predictions, test_predictions):
    if start_fold !=0:
        for fold in range(0,start_fold):
            fold_train_results, fold_test_results, fold_train_predictions, fold_test_predictions = load_fold_results(model_name, output_dir, static_str, fold)

            fold_train_results.loc['conf_matrix','0'] = eval(fold_train_results.loc['conf_matrix','0'], {"array": np.array, "int64": int})
            fold_test_results.loc['conf_matrix','0'] = eval(fold_test_results.loc['conf_matrix','0'], {"array": np.array, "int64": int})
            train_results.loc[fold] = fold_train_results['0']
            test_results.loc[fold] = fold_test_results['0']
            train_predictions.append(fold_train_predictions)
            test_predictions.append(fold_test_predictions)

def load_fold_results(model_name, output_dir, static_str, fold):
    results_dir = output_dir + rf"\{model_name}\cross-val{static_str}\results"
    fold_train_results = pd.read_csv(results_dir+ rf"\fold_{fold}_train_results.csv",index_col=0)
    fold_test_results = pd.read_csv(results_dir + rf"\fold_{fold}_test_results.csv",index_col=0)
    fold_train_predictions = pd.read_csv(results_dir + rf"\fold_{fold}_train_predictions.csv",index_col=0)
    fold_test_predictions = pd.read_csv(results_dir + rf"\fold_{fold}_test_predictions.csv",index_col=0)
    return fold_train_results,fold_test_results,fold_train_predictions,fold_test_predictions