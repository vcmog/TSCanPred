from datasets.utils.utils import (
    assign_pseudo_index,
    select_pre_index_events,
    remove_short_sequences_pandas,
    pad_lengths,
)
from datasets.utils.utils import build_event_sequences_list
from datetime import timedelta
import numpy as np


def generate_RNN_input(
    event_df,
    lead_time=0,
    observation_window=None,
    pad=True,
    zero_fill_nan=True,
    max_length=None,
    return_lengths=False,
    min_length=None,
):
    """
    Create array for input to an RNN model.
    Following steps:
    1. Assign pseudo index and limit to events occuring before the pseudo_index
    2. Build event sequences to turn events into a per-patient list of arrays
    3. Build outcomes df
    4. Build lengths
    5. If pad: pad inputs
    Optional steps: limit to observation period, zero-fill nans, filter out short sequences, zip lengths with inputs,
    pad sequences to max length.

    Args:
        event_df (DataFrame): Dataframe of format with columns patient_id, variable, event_dt,event_val, index_date
        lead_time (int, optional): Gap between index_date and prediction point in days.
        observation_winodw (int, optional): Maximum history used for prediction in days.
        pad (bool, optional): Whether to pad the input to maximum length. Defaults to True.
        zero_fill_nan (bool, optional): Whether to fill NaN values with zero or leave as NaN. Defaults to True.
    Returns:
        inputs, outcomes, ids
    """
    # Limit event_df to events that occur before the pseudo-index (one is assigned if needed).
    assign_pseudo_index(event_df, lead_time)
    event_df = select_pre_index_events(event_df)

    if (
        observation_window
    ):  # If observation window specified, limit dataframe to tests made within the window
        earliest_date = event_df["PSEUDO_INDEX"] - timedelta(days=observation_window)
        event_df = event_df[event_df["EVENT_DT"] >= earliest_date]

    patient_data = build_event_sequences_list(event_df)

    patient_ids = patient_data.index.to_numpy()

    outcomes = (
        event_df[["PATIENT_ID", "OUTCOME"]]
        .groupby("PATIENT_ID")
        .max()
        .loc[patient_ids]
        .to_numpy()
    )

    if zero_fill_nan:
        patient_data = patient_data.apply(lambda x: np.nan_to_num(x))

    lengths = np.array([len(x) for x in patient_data])
    if min_length is not None:
        patient_data, outcomes, lengths = remove_short_sequences_pandas(
            min_length, patient_data, outcomes, lengths
        )

    if pad:  # Pad with zeros to max length
        patient_data, padded_inputs = pad_lengths(max_length, patient_data)
        padded_inputs = np.array(padded_inputs)
        lengths = np.array(lengths).flatten()
        if return_lengths:
            inputs_with_lengths = np.array(
                list(zip(padded_inputs, lengths)), dtype=object
            )
            return inputs_with_lengths, outcomes, patient_ids
        else:
            return padded_inputs, outcomes, patient_ids

    else:
        return np.array(patient_data), outcomes, patient_ids
