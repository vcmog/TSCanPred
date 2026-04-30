import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from unused.feature_engineering import FeatureCleaner
from sklearn.linear_model import LassoCV
from features.transformers import SimpleImputerFeatureNames, StandardScalerFeatureNames


def build_feature_pipeline():
    fs = LassoCV(alphas=np.logspace(-4, 2, 100), cv=10, random_state=42)
    pipeline = Pipeline(
        [
            (
                "feature_cleaner",
                FeatureCleaner(
                    general_missing_threshold=0.9,
                    count_missing_threshold=0.99,
                    corr_threshold=0.85,
                    count_variable_pattern="count",
                ),
            ),
            (
                "imputer",
                SimpleImputerFeatureNames(missing_values=np.nan, strategy="mean"),
            ),
            ("scaler", StandardScalerFeatureNames()),
            ("feature_selector", SelectFromModel(fs, threshold="0.1*mean")),
        ]  # LinearSVC(penalty='l1', class_weight='balanced', C=10), threshold=1e-5))
    )
    return pipeline
