import torch


class MeanFillTransform:
    def __init__(self, precomputed_means, feature_dim=-1):
        self.means = precomputed_means
        self.feature_dim = feature_dim

    def __call__(self, x):
        feature_dim = (
            self.feature_dim if self.feature_dim >= 0 else x.ndim + self.feature_dim
        )

        expand_shape = [1] * x.ndim
        expand_shape[feature_dim] = -1
        means = self.means.view(*expand_shape)

        nan_mask = torch.isnan(x)
        filled_x = torch.where(nan_mask, means, x)
        return filled_x
