import numpy as np
import pandas as pd
from datetime import timedelta


def bin_measurements(event_df, freq="W", agg_method="mean"):
    """ "
    Divide sequence of a single measurement into bins. Aggregated by frequency given. Give 'differences'
    column indicating number of days between bin and index_date.

    Args:
        event_df (DataFrame): Data frame with an individual row for each measurement. Columns:
            - 'patient_id': Unique identifier for each patient.
            - 'variable':   Name of event.
            - 'event_val':      Numeric value of the measurement.
            - 'event_dt':   Date measurement was taken.
            - 'index_date': Index date of patients, e.g., recorded/determined date of outcome.
        freq (str):         Group measurements into bins of given frequency. Find possible ones here:
          <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>
        agg_method (str): Method with which to aggregate tests within the same bin. Must be valud method for Pandas .agg()

    Returns:
        DataFrame: Grouped data frame with the following columns:
            - 'patient_id': Unique identifier for each patient.
            - 'variable':
            - 'charttime
    """

    if "PSEUDO_INDEX" in event_df.columns:
        index_col = "PSEUDO_INDEX"
    else:
        index_col = "INDEX_DATE"

    custom_agg = {
        index_col: "first",  # Keep the first index dateas it should be the same for all entries
        "EVENT_VAL": "mean",  # Aggregate 'value' by finding the mean within that bin
    }

    # group into bins of given frequency
    grouped = (
        event_df[["PATIENT_ID", "VARIABLE", "EVENT_VAL", "EVENT_DT", index_col]]
        .groupby(["PATIENT_ID", "VARIABLE", pd.Grouper(key="EVENT_DT", freq=freq)])
        .agg(custom_agg)
        .reset_index()
    )
    # grouper operates using weekdays - 'weeks' are aligned at the start of the week, Sunday. Find a weekly
    # offset by finding the day of the week the index date is on and realigning the event dates. Avoids negative
    # weekly differences.
    index_date_day = event_df[index_col].dt.weekday
    grouped["period_offset"] = grouped["EVENT_DT"].dt.weekday - index_date_day
    grouped["EVENT_DT"] = grouped["EVENT_DT"] - pd.to_timedelta(
        grouped["period_offset"], unit="D"
    )
    grouped = grouped.drop("period_offset", axis=1)

    # find the differences between the time point and the index date in days
    grouped["differences"] = (
        grouped[index_col] - grouped["EVENT_DT"]
    )  # - third_level_data
    grouped["differences"] = grouped["differences"].apply(lambda x: x.days)
    return grouped  # .reset_index()


def get_time_diff_for_consecutive_tests(event_df):
    """Return differences between each test event_dt and the previous.

    Args:
        event_df (DataFrame): columns include 'PATIENT_ID', 'EVENT_DT'

    Returns:
        Series: Series with multilevel index ('PATIENT_ID', 'EVENT_DT') and the time_difference
          between consecutive rows in days.
    """
    # Add column 'time_diff' with difference between each row and the preceding
    event_df = event_df.sort_values(["PATIENT_ID", "EVENT_DT"])
    time_diff = event_df[["PATIENT_ID", "EVENT_DT"]].copy().drop_duplicates()
    time_diff["time_diff"] = time_diff.groupby("PATIENT_ID")["EVENT_DT"].diff()
    time_diff["time_diff"] = (
        time_diff["time_diff"].dt.total_seconds() / (24 * 3600)
    ).fillna(0)
    time_diff = time_diff.set_index(["PATIENT_ID", "EVENT_DT"])
    return time_diff
