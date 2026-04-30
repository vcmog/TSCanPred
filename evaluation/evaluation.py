import pandas as pd
import torch
from sklearn.metrics import (
    precision_score,
    average_precision_score,
    recall_score,
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import numpy as np
import matplotlib.pyplot as plt


def evaluate_performance_torchmodel(
    model, dataloader, save_dir=None, return_predictions=False, use_lengths=False
):
    model.eval()
    total_correct = 0
    total_samples = 0
    predicted_labels = []
    true_labels = []
    all_probabilities = []
    all_patient_ids = []
    with torch.no_grad():
        for batch in dataloader:

            inputs = batch[0]
            labels = batch[1]
            patient_ids = batch[2]
            static_data = batch[3]
            if len(batch) == 5:
                lengths = batch[4]

            outputs = model(inputs, static_data)
            probabilities = torch.sigmoid(outputs)

            predicted = (probabilities > 0.5).int()
            if predicted.ndim == 0:
                predicted = predicted.unsqueeze(0)
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)
            if probabilities.ndim == 0:
                probabilities = probabilities.unsqueeze(0)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            predicted_labels.extend(list(predicted.numpy()))
            true_labels.extend(list(labels.numpy()))
            all_probabilities.extend(list(probabilities.numpy()))
            all_patient_ids.extend(list(patient_ids.numpy()))

    predictions = pd.DataFrame(
        {
            "patient_id": all_patient_ids,
            "true_label": true_labels,
            "prediction": predicted_labels,
            "predicted_prob": all_probabilities,
        }
    )
    # save all predictions
    if save_dir:
        predictions.to_csv(save_dir)
    # Compute accuracy
    accuracy = total_correct / total_samples

    # Compute precision and recall
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Calculate false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(true_labels, all_probabilities)

    # Calculate AUROC
    auroc = roc_auc_score(true_labels, all_probabilities)
    avg_precision = average_precision_score(true_labels, all_probabilities)

    # Calculate balanced accuracy score
    balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)

    results = pd.Series(
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "conf_matrix": [conf_matrix],
            "auroc": auroc,
            "average_precision": avg_precision,
            "balanced_accuracy": balanced_accuracy,
        }
    )
    if return_predictions:
        return results, predictions
    else:
        return results


def evaluate_performance_GRUD(
    model, dataloader, save_dir=None, return_predictions=False, use_lengths=False
):
    model.eval()
    total_correct = 0
    total_samples = 0
    predicted_labels = []
    true_labels = []
    all_probabilities = []
    all_patient_ids = []
    with torch.no_grad():
        for batch in dataloader:

            inputs = batch[0]
            labels = batch[1]
            patient_ids = batch[2]
            static_data = batch[-1]
            outputs = model(inputs, static_data)
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).int()
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            predicted_labels.extend(list(predicted.numpy()))
            true_labels.extend(list(labels.numpy()))
            all_probabilities.extend(list(probabilities.numpy()))
            all_patient_ids.extend(list(patient_ids.numpy()))

    predictions = pd.DataFrame(
        {
            "patient_id": all_patient_ids,
            "true_label": true_labels,
            "prediction": predicted_labels,
            "predicted_prob": all_probabilities,
        }
    )
    # save all predictions
    if save_dir:
        predictions.to_csv(save_dir)
    # Compute accuracy
    accuracy = total_correct / total_samples

    # Compute precision and recall
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Calculate false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(true_labels, all_probabilities)

    # Calculate AUROC
    auroc = roc_auc_score(true_labels, all_probabilities)
    avg_precision = average_precision_score(true_labels, all_probabilities)

    # Calculate balanced accuracy score
    balanced_accuracy = balanced_accuracy_score(true_labels, predicted_labels)

    results = pd.Series(
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "conf_matrix": [conf_matrix],
            "auroc": auroc,
            "average_precision": avg_precision,
            "balanced_accuracy": balanced_accuracy,
        }
    )
    if return_predictions:
        return results, predictions
    else:
        return results


def evaluate_performance_sklearn_model(
    model, X, y_true, ids, threshold=0.5, save_dir=None, return_predictions=False
):
    """
    model:
    X
    y_true
    ids (np.array):
    """

    y_probs = model.predict_proba(X)[:, 1]
    y_pred = y_probs > threshold
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    results_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "conf_matrix": [conf_matrix],
        "auroc": auroc,
        "average_precision": avg_precision,
        "balanced_accuracy": balanced_accuracy,
    }

    predictions = pd.DataFrame(
        {
            "patient_id": ids.values,
            "true_label": y_true,
            "prediction": y_pred,
            "predicted_prob": y_probs,
        }
    )

    if save_dir:
        predictions.to_csv(save_dir)
    if return_predictions:
        return pd.Series(results_dict), predictions
    else:
        return pd.Series(results_dict)
