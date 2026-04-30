import pandas as pd


def get_features_at_lead_time(experiment_input_dir, use_static=False):
    # project_path = r'S:\1475 - Early detection of cancer occurrence and recurrence in primary care using ar\Data\Panc Cancer Prediction'
    input_path = experiment_input_dir

    blood_data = pd.read_csv(
        input_path + r"\cohort_bloods.csv",
        index_col=0,
        parse_dates=["PSEUDO_INDEX", "OBSERVATION_START", "EVENT_DT"],
    )
    variables = blood_data["VARIABLE"].unique()
    outcomes = (
        blood_data[["PATIENT_ID", "OUTCOME"]].drop_duplicates().set_index("PATIENT_ID")
    )
    blood_data["PREDICTION_POINT"] = blood_data["PSEUDO_INDEX"]
    blood_data["DAYS_BEFORE_PREDICTION"] = (
        blood_data["PREDICTION_POINT"] - blood_data["EVENT_DT"]
    ).dt.days

    blood_data_within_6 = blood_data[blood_data["DAYS_BEFORE_PREDICTION"] <= 180]
    blood_data_between_6_and_12 = blood_data[
        (
            (365 >= blood_data["DAYS_BEFORE_PREDICTION"])
            & (blood_data["DAYS_BEFORE_PREDICTION"] > 180)
        )
    ]

    agg_variables_proximal = get_agg_variables(
        blood_data_within_6, window_suffix="_proximal"
    )
    agg_variables_distal = get_agg_variables(
        blood_data_between_6_and_12, window_suffix="_distal"
    )

    all_variables = pd.merge(
        agg_variables_proximal,
        agg_variables_distal,
        right_index=True,
        left_index=True,
        how="outer",
    )
    all_variables.loc[
        :, [col for col in all_variables.columns if "count" in col]
    ] = all_variables.loc[
        :, [col for col in all_variables.columns if "count" in col]
    ].fillna(
        0
    )  # set missing counts to 0

    all_variables = get_trends(all_variables, variables)
    all_variables["OUTCOME"] = outcomes.loc[all_variables.index]["OUTCOME"]

    # Load demographic data
    if use_static == True:
        demographics = pd.read_csv(input_path + "\demographics.csv")
        all_variables = all_variables.merge(
            demographics, left_index=True, right_on="PATIENT_ID"
        ).set_index("PATIENT_ID")

    print("Number of features:", len(all_variables.columns) - 1)
    print(
        "Patients in original data:",
        blood_data["ALF_PE"].nunique(),
        "\nPatients with data in agg dataframe:",
        len(all_variables),
    )
    return all_variables


def get_agg_variables(blood_data_within_window, window_suffix):
    agg_variables_in_window = (
        blood_data_within_window[["PATIENT_ID", "VARIABLE", "EVENT_VAL"]]
        .groupby(["PATIENT_ID", "VARIABLE"])
        .agg(["mean", "count"])
        .add_suffix(window_suffix)
        .reset_index()
    )  #'max', 'min',
    agg_variables_pivot_in_window = agg_variables_in_window.pivot(
        index="PATIENT_ID", columns="VARIABLE"
    )
    agg_variables_pivot_in_window.columns = [
        f"{col[2]}_{col[1]}" for col in agg_variables_pivot_in_window.columns
    ]
    return agg_variables_pivot_in_window


def get_trends(df, variables):
    for variable in variables:
        df[f"{variable}_trend"] = (
            df[f"{variable}_mean_distal"] - df[f"{variable}_mean_proximal"]
        )
    return df
