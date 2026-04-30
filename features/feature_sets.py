FEATURE_SETS = {
    "all": lambda cols:cols,
    "proximal_means": lambda cols: [c for c in cols if c.endswith("_mean_proximal")]
}

def select_features(X, feature_set_name="all"):
    if feature_set_name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set_name}")
    selector = FEATURE_SETS[feature_set_name]
    selected_columns = selector(X.columns)
    return X[selected_columns]