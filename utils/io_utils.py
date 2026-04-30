import os
import config
import matplotlib.pyplot as plt


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def make_crossval_directories(lead_time, static_str, model_name, config_file, feature_set_name=""):
    _, _, output_dir, model_dir = config.get_dirs(config_file)
    model_dir = model_dir + rf"\lead_time={lead_time}"
    ensure_dir_exists(model_dir)

    output_dir = output_dir + rf"\lead_time={lead_time}"
    ensure_dir_exists(output_dir)

    model_specific_output_dir = output_dir + rf"\{model_name}"
    ensure_dir_exists(model_specific_output_dir)

    cross_val_experiement_dir = model_specific_output_dir + rf"\cross-val{static_str}{feature_set_name}"
    ensure_dir_exists(cross_val_experiement_dir)

    results_dir = cross_val_experiement_dir+r"\results"
    ensure_dir_exists(results_dir)

    predictions_dir = cross_val_experiement_dir+r"\predictions"
    ensure_dir_exists(predictions_dir)
    return model_dir, output_dir

def save_fold_results(model_name, output_dir, static_str, fold, fold_train_results, fold_train_predictions, fold_test_results, fold_test_predictions):
    fold_train_results.to_csv(output_dir + rf"\{model_name}\cross-val{static_str}\results\fold_{fold}_train_results.csv")
    fold_test_results.to_csv(output_dir + rf"\{model_name}\cross-val{static_str}\results\fold_{fold}_test_results.csv")
    fold_train_predictions.to_csv(output_dir + rf"\{model_name}\cross-val{static_str}\results\fold_{fold}_train_predictions.csv")
    fold_test_predictions.to_csv(output_dir + rf"\{model_name}\cross-val{static_str}\results\fold_{fold}_test_predictions.csv")
    
def save_training_curve(training_losses, val_losses, dir):
    """
    Given the training and validation losses, plot the training curves.
    """
    plt.figure(figsize=(7, 4), dpi=600)
    plt.plot(training_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(dir)
