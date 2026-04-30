## for general data-related utils, i.e., not specific to my dataset and could be reused

import torch


def count_outcomes_from_dataloader(loader, label_index=1):
    """
    Take a dataloader and count the number of patients
    and the number of patients with the positive class.

    Args:
        loader: (DataLoader)

    Returns:
        n_p_class (int): Number of patients with the positive class.
        n_patients
    """
    n_p_class = 0
    n_patients = 0
    for batch in loader:
        labels = batch[label_index]
        n_p_class += sum(labels)
        n_patients += len(labels)
    return n_p_class, n_patients


def compute_balanced_class_weights(outcomes):
    """Same formula as the sckitlearn compute_class_weight with 'balanced'"""
    return torch.tensor(
        [
            len(outcomes) / (2 * sum(outcomes == 1)),
            len(outcomes) / (2 * sum(outcomes == 0)),
        ]
    )
