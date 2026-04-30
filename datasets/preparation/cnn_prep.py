import pandas as pd
import numpy as np
from datasets.utils import time_utils


def generate_CNN_input(
    event_df, lead_time=0, observation_window=None, bins="W", max_len=None
):
    """ "
    Create a 3D array for Convolutional Neural Network (CNN) input.

    Args:
        event_df (DataFrame): Laboratory events dataframe with columns patient_id, variable, event_dt,event_val, index_date, and outcome.
        lead_time (int, optional): Gap between index_date and prediction point in days.
        observation_winodw (int, optional): Maximum history used for prediction in days.
        bins (str, optional): What bins to use for the events, default weekly. Alternative value is 'ME', monthly based on 'MONTH_END'.
        max_len (int): If set, this is the maximum number of bins in the dataset.
    Returns:
        array_3d (ndarray): 3D array with dimensions (subject_id, variable, time_difference)
    """
    valid_bins = ["W", "ME"]
    if bins not in valid_bins:
        raise ValueError(f"Invalid value: {bins}. Must be one of {valid_bins}")
    if bins == "W":
        bin_divisor = 7
    else:
        bin_divisor = 30
    # bin measurements and find differences in weeks
    binned_df = time_utils.bin_measurements(event_df, freq=bins)
    binned_df["bin_differences"] = round(binned_df["differences"] // bin_divisor, 0)

    # restrict to measurements before lead_time and within the observation window.
    if lead_time:
        binned_df = binned_df[binned_df["differences"] > lead_time]
    if observation_window:
        binned_df = binned_df[binned_df["differences"] < observation_window]

    pivot_df = (
        binned_df.reset_index()
        .fillna(0)
        .pivot_table(
            index=["PATIENT_ID", "bin_differences"],
            columns="VARIABLE",
            values="EVENT_VAL",
        )
    )

    # find the outcome for patients in the CNN dataframe, preserving the order of the patient_ids
    patient_ids = pivot_df.index.get_level_values(0).drop_duplicates()
    outcome = (
        event_df[["PATIENT_ID", "OUTCOME"]]
        .drop_duplicates()
        .set_index("PATIENT_ID")
        .loc[patient_ids]["OUTCOME"]
    ).to_numpy()
    # fill missing values with 0
    pivot_df = pivot_df.fillna(0)

    # Get all possibles differences
    if lead_time:
        lead_time = round(lead_time // bin_divisor, 0)
    # TODO: add the upper end of this range so it's not dependent on the len of the data
    if max_len:
        all_differences = range(lead_time, max_len)
    else:
        all_differences = range(lead_time, max(binned_df["bin_differences"]) + 1)

    # Create a multiindex with all possible combinations of difference and subject_id
    multiindex = pd.MultiIndex.from_product(
        [pivot_df.index.levels[0], all_differences], names=["PATIENT_ID", "differences"]
    )

    # Reindex the pivot table to ensure all combinations are present, filling missing values with NaN
    pivot_df = pivot_df.reindex(multiindex)

    # 3D array with dimensions (patient_id, variable, bin)
    array_3d = pivot_df.values.reshape(
        (-1, len(pivot_df.columns), pivot_df.index.get_level_values(1).nunique())
    )

    array_3d = np.nan_to_num(array_3d)

    return array_3d, outcome, patient_ids.to_numpy()
