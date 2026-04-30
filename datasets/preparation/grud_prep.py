import numpy as np
from datasets.utils.utils import (
    assign_pseudo_index,
    select_pre_index_events,
    build_sequences_to_array,
)
from datetime import timedelta
from datasets.utils.time_utils import get_time_diff_for_consecutive_tests
import torch


def generate_GRUD_array(event_df, max_len, lead_time=0, observation_window=None):
    """
    Takes event_df and returns np.arrays of data, outcomes, and patient_ids.
    Args:
        event_df (DataFrame): DataFrame of blood tests with columns PATIENT_ID, EVENT_DT, VARIABLE,
        EVENT_VAL, and OUTCOME. Also either 'INDEXDATE' or 'PSUEDO_INDEX'.
    Returns:
        patient_data (np.array): (Batch, time_step, features)
        outcomes (np.array): labels for patients.
        ids (np.array): patient_ids, corresponding with order of patient_data and outcomes.
    """
    # Unique ids
    patient_ids = event_df["PATIENT_ID"].unique().to_numpy()

    # # Limit event_df to events that occur before the pseudo-index
    assign_pseudo_index(event_df, lead_time)
    event_df = select_pre_index_events(event_df)

    if observation_window:  # limit to events within given window before index
        event_df = event_df[
            event_df["EVENT_DT"]
            >= event_df["PSEUDO_INDEX"] - timedelta(days=observation_window)
        ]

    # Add column 'time_diff' with difference between each row and the preceding
    time_diff = get_time_diff_for_consecutive_tests(event_df)

    # pivot the dataframe so that each variable is a column
    pivoted_df = event_df.pivot_table(
        index=["PATIENT_ID", "EVENT_DT"], columns="VARIABLE", values="EVENT_VAL"
    )

    # add in the time-diff column
    pivoted_df = pivoted_df.merge(time_diff, on=["PATIENT_ID", "EVENT_DT"], how="left")
    cols = list(pivoted_df.columns)  # ensure time_diff is the last column
    cols.append(cols.pop(cols.index("time_diff")))
    pivoted_df = pivoted_df[cols]

    outcomes = (
        event_df[["PATIENT_ID", "OUTCOME"]].groupby("PATIENT_ID").max().loc[patient_ids]
    )
    outcomes = np.array(outcomes).flatten()

    X = build_sequences_to_array(max_len, patient_ids, pivoted_df, cols)

    return X, outcomes, patient_ids


def generate_delta_mask_sequence(
    data, train_means=None, train_stds=None, max_delta=None, return_descriptors=False
):
    X = data[:, :, :-1]  # select blood test columns, exclude time_diff
    mask = ~np.isnan(X)  # 1 if measurement present, 0 if missing.

    # standardize across patients and timesteps
    if train_means is None:
        train_means = np.nanmean(X, axis=(0, 1))
    if train_stds is None:
        train_stds = np.nanstd(X, axis=(0, 1))
    X = (X - train_means) / train_stds

    # Time_lags is S vector in paper -> contains time lags of all examples (N x Time_steps)
    time_diff = data[:, :, -1]  # the last column in the feature vector
    time_lags = time_diff.copy()
    time_lags[:, 0] = 0
    # Get lags from first index. Use :1 for as first index will be 0 for all entries.

    # Find the lengths of all time series
    lengths = ((time_lags) != 0).sum(axis=1)

    Delta = np.repeat(
        time_lags[:, :, None], X.shape[2], axis=2
    )  # repeat patient delta for every feature

    X_last_observed, Delta = forward_fill_and_accumulate_delta(
        train_means, X, mask, Delta
    )

    if not max_delta:
        max_delta = np.nanmax(Delta)
    if max_delta > 0:
        Delta = Delta / max_delta

    # Expand dimensions to prepare for concatenation
    X = np.expand_dims(X, axis=1)
    X_last_observed = np.expand_dims(X_last_observed, axis=1)
    mask = np.expand_dims(mask, axis=1)
    Delta = np.expand_dims(Delta, axis=1)
    dataset_combined = np.concatenate((X, X_last_observed, mask, Delta), axis=1)

    X_mean = torch.Tensor(np.nanmean(X, axis=0))
    if return_descriptors:
        return dataset_combined, lengths, X_mean, train_means, train_stds, max_delta

    else:
        return dataset_combined, lengths


def forward_fill_and_accumulate_delta(train_means, X, mask, Delta):
    _, n_timesteps, _ = X.shape
    X_last_observed = np.copy(X)
    # --- Step 1: fill first value of last_observed
    X_last_observed[:, 0, :] = np.where(
        mask[:, 0, :] == 1,
        X_last_observed[:, 0, :],  # keep observed
        train_means,  # else replace with mean
    )
    # --- Step 2: forward-fill + accumulate deltas in one loop
    for t in range(1, n_timesteps):
        # forward fill missing values
        X_last_observed[:, t, :] = np.where(
            mask[:, t, :] == 1,
            X_last_observed[:, t, :],  # keep observed
            X_last_observed[:, t - 1, :],  # Forward-fill last value
        )
        # accumulate deltas if previous timestep was missing
        Delta[:, t, :] += (mask[:, t - 1, :] == 0) * Delta[:, t - 1, :]
    return X_last_observed, Delta
