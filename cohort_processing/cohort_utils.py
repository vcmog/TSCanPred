import pandas as pd
from tqdm import tqdm


def find_valid_blocks(block_df, lead_time, observation_window):
    """
    Given the lead time and observation window, find the valid blocks.

    Args:
        block_df (DataFrame):      Dataframe of continuous blocks. Columns:
            - patient_id:               Unique identifier for patients.
            - outcome:                  Binary outcome, 1 for the positive case and 0 for the negative.
            - indexdate:                Date where the outcome is determined.
            - record_start:             Beginning of the registration block.
            - block_end:                End of the registration period.
            - WOB:                      Date of birth.
            - GNDR_CD:                  Binary indicator, 0 is male, 1 is female.
            - block_end_to_index:       The difference between the block_end and the index_date. If negative,
                                        the block_end is after the index_date.
            - record_start_to_index:    Days between the record_start and the index_date.
            - age_at_index:             Age at the index date.
            - index_year:               Year of the index date.
            - block_id:                 Unique identifier for the registration block.
        lead_time (int):
    """

    # min length of record required for valid blocks
    min_record_len = observation_window + lead_time

    # restrict to blocks where the gap between the block_end and index is less than equal to
    # index, and the record_start to index is at least as long as the min_record_len
    valid_blocks = block_df[
        (block_df["BLOCK_END_TO_INDEX"] <= lead_time)
        & (block_df["RECORD_START_TO_INDEX"] >= min_record_len)
    ]

    return valid_blocks


def match_age_and_gender(block_df, n_matches, age_tolerance=3):
    """
    Match controls to cases based on sex and gender.

    Args:
        block_df (DataFrame):
            - patient_id:               Unique identifier for patients.
            - outcome:                  Binary outcome, 1 for the positive case and 0 for the negative.
            - indexdate:                Date where the outcome is determined.
            - record_start:             Beginning of the registration block.
            - block_end:                End of the registration period.
            - WOB:                      Date of birth.
            - GNDR_CD:                  Binary indicator, 0 is male, 1 is female.
            - block_end_to_index:       The difference between the block_end and the index_date.
                                        If negative, the block_end is after the index_date.
            - record_start_to_index:    Days between the record_start and the index_date.
            - age_at_index:             Age at the index date.
            - index_year:               Year of the index date.
            - block_id:                 Unique identifier for the registration block.
        n_matches (int):        Number of controls to match to a case
        age_tolerance (int):    Matches must be within this margin in age.
    """

    # seperate dataframe into cases and controls
    case_atts = block_df[block_df["OUTCOME"] != 0]
    print(len(case_atts))
    control_atts = block_df[block_df["OUTCOME"] == 0]
    print(len(control_atts))
    # initialise match_id (for analysing matches) and empty list for complete matches
    match_id = 0
    matched_records = []

    for gender in [0, 1]:
        case = case_atts[case_atts["GNDR_CD"] == gender]
        controls = control_atts[control_atts["GNDR_CD"] == gender]

        print("Matching gender", gender)
        # Iterate through cases. tqdm prints progress bar.
        for i, row in tqdm(case.iterrows()):
            age = row["AGE_AT_INDEX"]
            # control_close are controls within the age tolerance
            control_close = controls[controls["AGE_AT_INDEX"] >= age - age_tolerance]
            control_close = control_close[
                control_close["AGE_AT_INDEX"] <= age + age_tolerance
            ]

            # find the closest controls in age
            age_diff = abs(control_close["AGE_AT_INDEX"] - age)
            matches = age_diff.nsmallest(n_matches).index

            # append case to matched_records
            matched_records.append(
                {
                    "PATIENT_ID": row["PATIENT_ID"],
                    "AGE_AT_INDEX": row["AGE_AT_INDEX"],
                    "RECORD_START": row["RECORD_START"],
                    "BLOCK_END": row["BLOCK_END"],
                    "BLOCK_ID": row["BLOCK_ID"],
                    "GNDR_CD": row["GNDR_CD"],
                    "OUTCOME": 1,
                    "MATCH_ID": match_id,
                    "INDEXDATE": row["INDEXDATE"],
                }
            )

            # append closest controls to matched_records
            rows = control_close.loc[matches][
                [
                    "PATIENT_ID",
                    "AGE_AT_INDEX",
                    "RECORD_START",
                    "BLOCK_END",
                    "BLOCK_ID",
                    "GNDR_CD",
                    "INDEXDATE",
                ]
            ]
            rows["OUTCOME"] = 0
            rows["MATCH_ID"] = match_id
            matched_records += rows.to_dict(orient="records")

            # remove matched controls from the matching pool
            controls = controls[~controls["PATIENT_ID"].isin(rows["PATIENT_ID"])]

            match_id += 1

    return pd.DataFrame(matched_records)
