import numpy as np
import pandas as pd
import torch
from datetime import timedelta
from datasets.samplers import BatchSampler
from datasets.wrappers import WrapperDataset, NormalizeGRUDDatasetWrapper
from torch.utils.data import DataLoader
from features.preprocessing import get_age_mean_std, get_mean_std_from_time_data
from torchvision import transforms
from features.transformers import NormalizeAge, NormalizeData
from datasets.transforms import MeanFillTransform
from typing import Literal
from datasets.utils.time_utils import get_time_diff_for_consecutive_tests

def trim_long_sequences(data, lengths, max_length):
    X = []
    for idx in range(len(data)):
        start = 0
        end = int(lengths[idx])
        if max_length is not None and (end - start) > max_length:
            end = start + max_length
        X.append(torch.Tensor(data[idx, :, start:end, :]))
    return X


def remove_short_sequences_pandas(min_length, patient_data, outcomes, lengths):
    mask = np.where(lengths >= min_length)
    patient_data = patient_data.iloc[mask]
    outcomes = outcomes[mask].flatten()
    lengths = lengths[mask].flatten()
    return patient_data, outcomes, lengths


def remove_short_sequences_numpy(data, labels, lengths, dems, length_required):
    mask = np.where(lengths >= length_required)
    if not mask:
        print("Warning! No patients have trajectories of sufficient length.")

    data = data[mask]
    labels = labels[mask]
    lengths = lengths[mask]
    dems = dems[mask]
    return data, labels, lengths, dems


def get_loader_with_batch_sampling(
    dataset, batch_size=64, num_bins=2, collate_fn=None, num_workers=0
):
    lengths = [sample[3] for sample in dataset]
    batches = BatchSampler(lengths, batch_size, num_bins=num_bins)
    loader = DataLoader(
        dataset, batch_sampler=batches, collate_fn=collate_fn, num_workers=num_workers
    )
    return loader


def dataset_split(dataset, proportion=0.8):
    """
    Split dataset into 2 parts, with ratio proportion:(1-proportion).

    Args:
        dataset (Torch.Dataset): Dataset to be divided
        proportion (float): Between 0 and 1, the proportion to be assigned
        to the majority set.

    Returns:
        majority_set (torch.Dataset): Dataset of length proportion*original length.
        minority_set (torch.Dataset): Dataset containing remaining datapoints.
    """
    majority_size = int(proportion * len(dataset))
    minority_size = len(dataset) - majority_size

    majority_set, minority_set = torch.utils.data.random_split(
        dataset,
        [majority_size, minority_size],
        generator=torch.Generator().manual_seed(42),
    )
    return majority_set, minority_set


def apply_transforms(
    train_set,
    test_set,
    variable_dim,
    model_type: Literal["RNN", "GRUD"] = "RNN",
    val_set=None,
):
    if model_type == "RNN":
        mean, std = get_mean_std_from_time_data(train_set, variable_dim)
        wrapper = WrapperDataset
    elif model_type == "GRUD":
        mean, std = get_mean_std_from_GRUD_time_data(train_set, variable_dim)
        wrapper = NormalizeGRUDDatasetWrapper

    train_age_mean, train_age_std = get_age_mean_std(train_set)

    # Set transfroms and apply to datasets using wrapper datasets
    custom_transform = transforms.Compose(
        [
            MeanFillTransform(mean, feature_dim=(variable_dim - 1)),
            NormalizeData(mean=mean, std=std, feature_dim=(variable_dim - 1)),
        ]
    )
    static_data_transform = NormalizeAge(train_age_mean, train_age_std)

    train_set = wrapper(train_set, custom_transform, static_data_transform)
    test_set = wrapper(test_set, custom_transform, static_data_transform)
    if val_set:
        val_set = wrapper(val_set, custom_transform, static_data_transform)
        return train_set, test_set, val_set

    return train_set, test_set


def assign_pseudo_index(event_df, lead_time):
    if "PSEUDO_INDEX" not in event_df.columns:
        if lead_time:
            event_df["PSEUDO_INDEX"] = event_df["INDEX_DATE"] - timedelta(
                days=lead_time
            )
        else:
            event_df["PSEUDO_INDEX"] = event_df["INDEX_DATE"]


def select_pre_index_events(event_df):
    event_df = event_df[event_df["EVENT_DT"] <= event_df["PSEUDO_INDEX"]].copy()
    return event_df


def pad_lengths(patient_data, max_length=None):
    """
    Given a df with rows
    """
    if max_length:
        max_len = max_length
        patient_data = patient_data.apply(
            lambda x: x[-max_len:] if len(x) > max_len else x
        )
    else:
        max_len = max([len(x) for x in patient_data])

    padded_inputs = patient_data.apply(
        lambda x: np.pad(
            x, ((max_len - len(x), 0), (0, 0)), mode="constant", constant_values=0
        )
    )
    # shape in form (patient_id, timesteps, features)
    padded_inputs = np.dstack(padded_inputs.values).transpose((2, 0, 1))
    return patient_data, padded_inputs


def fill_na_zero(df):
    """
    Fill missing values in the DataFrame with zeroes

    Args:
        df (DataFrame): The dataframe with missing values zero-imputed.
    """

    for col in df.columns:
        if col != "PATIENT_ID":
            df[col].fillna(0, inplace=True)
    return df


def fill_na_mean(df):
    """
    Fill missing values in the DataFrame with the mean of each column, or zero for trend features.

    Args:
        df (DataFrame): The dataframe with missing values mean-imputed.
    """

    for col in df.columns:
        if "trend" in col:
            df[col] = df[col].fillna(0)
        elif col != "PATIENT_ID":
            df[col] = df[col].fillna(df[col].mean(skipna=True))
    return df


missing_data_dispatcher = {"mean": fill_na_mean, "zero": fill_na_zero}


def get_mean_std_from_GRUD_time_data(dataset, variable_dim):
    """Variable dim is the feature dimension."""
    # Compute the mean and standard deviation of the dataset
    mean_sum = 0.0
    sum_of_squares = 0.0
    valid_count = 0

    for item in dataset:
        input_data = item[0]
        data = input_data[0, :, :]
        valid_mask = ~torch.isnan(data)
        reduced_dims = tuple(i for i in range(data.ndim) if i != (variable_dim - 1))
        # (variable dim is dim we want to keep, but is from shape assuming batch at dim 0, which isn't case here)
        mean_sum += data.nansum(
            dim=reduced_dims
        )  # Calculate mean along batch (0), and time_step dimension
        valid_count += valid_mask.sum(dim=reduced_dims)
        data_squared = data**2
        sum_of_squares += torch.nansum(data_squared, dim=reduced_dims)

    mean = mean_sum / valid_count
    variance = (sum_of_squares / valid_count) - (mean**2)
    std = torch.sqrt(variance)
    return mean, std


def get_age_mean_std(dataset):
    mean_sum = 0.0
    sum_of_squares = 0.0
    valid_count = 0
    for item in dataset:
        static_data = item[3]
        mean_sum += static_data[0]
        valid_count += 1
        sum_of_squares += static_data[0] ** 2

    mean = mean_sum / valid_count
    variance = (sum_of_squares / valid_count) / (mean**2)
    std = torch.sqrt(variance)
    return mean, std


def get_mean_std_from_time_data(dataset, variable_dim):
    """Variable dim is the feature dimension."""
    # Compute the mean and standard deviation of the dataset
    mean_sum = 0.0
    sum_of_squares = 0.0
    valid_count = 0

    for item in dataset:
        data = item[0]
        valid_mask = ~torch.isnan(data)
        reduced_dims = tuple(i for i in range(data.ndim) if i != (variable_dim - 1))
        # (variable dim is dim we want to keep, but is from shape assuming batch at dim 0, which isn't case here)
        mean_sum += data.nansum(
            dim=reduced_dims
        )  # Calculate mean along batch (0), and time_step dimension
        valid_count += valid_mask.sum(dim=reduced_dims)
        data_squared = data**2
        sum_of_squares += torch.nansum(data_squared, dim=reduced_dims)

    mean = mean_sum / valid_count
    variance = (sum_of_squares / valid_count) - (mean**2)
    std = torch.sqrt(variance)
    return mean, std


def get_dem_variables(experiment_input_dir, ids):
    dems = pd.read_csv(experiment_input_dir + r"\demographics.csv", index_col=0)
    dems = dems.loc[ids]

    return dems.values


def get_train_test_dem_variables(experiment_input_dir, train_ids, test_ids):
    dem_sets = []
    for subset, ids in [("train", train_ids), ("test", test_ids)]:
        dems = pd.read_csv(
            experiment_input_dir + rf"\{subset}_demographics.csv", index_col=0
        )
        dem_sets += [dems.loc[ids]]

    return dem_sets[0], dem_sets[2]


### --- lower level --- ###
def build_event_sequences_list(event_df):
    """
    Transform raw long-format event data into per-ID sequences of feature vectors.

    Performs the following steps:
    1. Computes time differences between consecutive events for each PATIENT_ID.
    2. Pivots the event data so that each VARIABLE becomes a column with EVENT_VAL as the value.
    3. Merges the pivoted data with the time differences.
    4. Groups the resulting data by ID and converts each group into a list on NumPy row vectors.
    """
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
    # Create a list of arrays for each patient of their events
    patient_data = pivoted_df.groupby("PATIENT_ID").apply(
        lambda x: list(x.to_numpy()), include_groups=False
    )
    return patient_data


def build_sequences_to_array(max_len, patient_ids, pivoted_df, cols):
    n_patients = len(patient_ids)
    n_cols = len(cols)
    X = np.zeros((n_patients, max_len, n_cols))

    for idx, patient in enumerate(patient_ids):
        data = pivoted_df.loc[
            patient
        ].reset_index()  # reset_index so can directly index
        length = len(data)

        if length <= max_len:
            X[idx, :length, :] = data.loc[0:length, cols].to_numpy()
        else:
            X[idx, :, :] = data.loc[length - max_len : length, cols].to_numpy()
    return X


#def build_feature_mapper():
 #   mapper_df = pd.read_csv(r'..\..\utils\labtestmapper.csv', index_col=0)
  #  mapper = mapper_df.to_dict()['category']
   # return mapper

#mapper = build_feature_mapper()


#def map_feature_panel(feature):
   # panel = mapper[feature]
    #
    # 
    # return panel