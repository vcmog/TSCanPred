import pandas as pd
import numpy as np
from features.preparation import build_feature_pipeline
from utils.results import save_results_dfs, build_results_structures
from hyperparameter_tune.hyperparameter_tune import hyperparameter_tune_sklearn_model
from models.model_dispatcher import model_dispatcher
from evaluation.evaluation import evaluate_performance_sklearn_model
from utils.io_utils import make_crossval_directories
from utils.splits import get_kfold_split_indices, get_datasplits


def nested_cross_val(
    model_name,
    X,
    y,
    lead_time,
    config_file,
    n_inner_trials=50,
    existing_best_params=None,
    use_static=False,
    feature_set_name="",
):

    
    (
    train_results,
    test_results,
    train_predictions,
    test_predictions,
    feature_use_counter,
    all_importances,
    ) = build_results_structures(get_feature_use_counter=True, get_importances=True)

    splits = get_kfold_split_indices(X, y, random_state=42)

    for n_fold, (fold_train, fold_test) in enumerate(splits):

        X_train, X_test, y_train, y_test, train_ids, test_ids = get_datasplits(
            X=X, y=y, fold_train=fold_train, fold_test=fold_test
        )

        (
            fold_train_results,
            fold_train_predictions,
            fold_test_results,
            fold_test_predictions,
            feature_names,
            importances,
        ) = run_cv_fold(
            model_name,
            n_inner_trials,
            existing_best_params,
            n_fold,
            X_train,
            X_test,
            y_train,
            y_test,
            train_ids,
            test_ids,
        )

        train_results.loc[n_fold] = fold_train_results
        test_results.loc[n_fold] = fold_test_results
        train_predictions.append(fold_train_predictions)
        test_predictions.append(fold_test_predictions)
        all_importances[n_fold] = importances
        feature_use_counter.update(feature_names)

    train_predictions = pd.concat(train_predictions)
    test_predictions = pd.concat(test_predictions)
    print(test_results)
    feature_use_counter = pd.DataFrame.from_dict(feature_use_counter, orient="index")
    if not model_name == "NN":
        importances = pd.concat([all_importances[i].T for i in range(0, 5)])
    else:
        importances = None

    if use_static == True:
        static_str = "_with_static"
    else:
        static_str = ""

    if feature_set_name!='all':
        feature_set_name = "_"+feature_set_name
    else:feature_set_name==""
    model_dir, output_dir = make_crossval_directories(
        lead_time, static_str=static_str, model_name=model_name, feature_set_name=feature_set_name, config_file=config_file
    )
    print(output_dir)
    save_results_dfs(
        model_name=model_name,
        train_results=train_results,
        test_results=test_results,
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        feature_use_counter=feature_use_counter,
        importances=importances,
        static_str=static_str,
        output_dir=output_dir,
        feature_names=feature_names,
        feature_set_name=feature_set_name
    )
    return (
        train_results,
        train_predictions,
        test_results,
        test_predictions,
        feature_use_counter,
    )


def run_cv_fold(
    model_name,
    n_inner_trials,
    existing_best_params,
    n_fold,
    X_train,
    X_test,
    y_train,
    y_test,
    train_ids,
    test_ids,
):
    pipeline = build_feature_pipeline()

    X_train_processed = pipeline.fit_transform(X_train, y_train)
    X_test_processed = pipeline.transform(X_test)
    print(f"Fold {n_fold} no. features:", len(pipeline.get_feature_names_out()))

    if existing_best_params is None:
        best_params = hyperparameter_tune_sklearn_model(
            model_name, X_train_processed, y_train, n_trials=n_inner_trials
        )
    else:
        best_params = existing_best_params
    model = model_dispatcher[model_name](**best_params)

    model.fit(X_train_processed, y_train)

    fold_train_results, fold_train_predictions = evaluate_performance_sklearn_model(
        model=model,
        X=X_train_processed,
        y_true=y_train,
        ids=train_ids,
        return_predictions=True,
    )
    fold_test_results, fold_test_predictions = evaluate_performance_sklearn_model(
        model=model,
        X=X_test_processed,
        y_true=y_test,
        ids=test_ids,
        return_predictions=True,
    )
    fold_train_results["fold"] = fold_train_predictions["fold"] = fold_test_results[
        "fold"
    ] = fold_test_predictions["fold"] = n_fold

    print("Model evaluated.")

    feature_names = pipeline.get_feature_names_out()
    if not model_name == "NN":
        importances = pd.DataFrame(
            np.abs(model.coef_.reshape(-1)), index=feature_names, columns=["importance"]
        ).sort_values("importance", ascending=False)
        print(importances)  # .index)
        features = importances.index[np.where(importances > 0)[0]]
        print(f"Features used (n={sum(importances['importance']>0)}): {features}")
    else:
        importances = None
    return (
        fold_train_results,
        fold_train_predictions,
        fold_test_results,
        fold_test_predictions,
        feature_names,
        importances,
    )
