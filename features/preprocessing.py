import torch


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
