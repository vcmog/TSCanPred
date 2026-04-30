import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class NormalizeAge:
    def __init__(self, mean_age, std_age):
        self.mean_age = mean_age
        self.std_age = std_age

    def __call__(self, age):
        normalized_age = (age - self.mean_age) / self.std_age
        return normalized_age


class NormalizeData:
    def __init__(self, mean, std, feature_dim=-1):
        self.mean = mean
        self.std = std
        self.feature_dim = feature_dim

    def __call__(self, tensor):
        feature_dim = (
            self.feature_dim
            if self.feature_dim >= 0
            else tensor.ndim + self.feature_dim
        )
        expand_shape = [1] * tensor.ndim
        expand_shape[feature_dim] = -1
        means = self.mean.view(*expand_shape)
        stds = self.std.view(*expand_shape)
        return (tensor - means) / stds


class SimpleImputerFeatureNames(SimpleImputer):

    def fit(self, X, y=None):
        self.feature_names_ = X.columns
        return super().fit(X, y)

    def transform(self, X):
        imputed_array = super().transform(X)
        return pd.DataFrame(imputed_array, columns=self.feature_names_)


class StandardScalerFeatureNames(StandardScaler):

    def fit(self, X, y=None):
        self.feature_names_ = X.columns
        return super().fit(X, y)

    def transform(self, X):
        imputed_array = super().transform(X)
        return pd.DataFrame(imputed_array, columns=self.feature_names_)
