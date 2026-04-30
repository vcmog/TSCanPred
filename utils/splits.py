from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np


def get_kfold_split_indices(X, y, n_splits=5, random_state=41):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = cv.split(X, y)
    return splits


def get_datasplits(X, y, fold_train, fold_test, patient_ids=None, all_dems=None):

    train_ids = test_ids = None  # will get overwritten if present

    if isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[fold_train], X.iloc[fold_test]
        train_ids = X_train.index
        test_ids = X_test.index
    elif isinstance(X, np.ndarray):
        X_train, X_test = X[fold_train], X[fold_test]
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y_train, y_test = y.iloc[fold_train], y.iloc[fold_test]
    elif isinstance(y, np.ndarray) or isinstance(y, list):
        y_train, y_test = y[fold_train], y[fold_test]

    if patient_ids is not None:
        if isinstance(patient_ids, np.ndarray):
            train_ids, test_ids = patient_ids[fold_train], patient_ids[fold_test]
        elif isinstance(patient_ids, pd.DataFrame):
            train_ids, test_ids = (
                patient_ids.iloc[fold_train],
                patient_ids.iloc[fold_test],
            )
    return_items = [X_train, X_test, y_train, y_test, train_ids, test_ids]

    if all_dems is not None:
        if isinstance(all_dems, np.ndarray):
            return_items.append(all_dems[fold_train])
            return_items.append(all_dems[fold_test])

        elif isinstance(all_dems, pd.DataFrame):
            return_items.append(all_dems.iloc[fold_train])
            return_items.append(all_dems.iloc[fold_test])

    return return_items
