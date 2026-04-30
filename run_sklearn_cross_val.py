from features.builders import get_features_at_lead_time
from training.sklearn.cross_validation import nested_cross_val
import config 
from features.feature_sets import select_features
import argparse

if __name__ == "__main__":
    # initialise argument parser
    parser = argparse.ArgumentParser()

    # add arguments to parse
    parser.add_argument(
        "--model", type=str, choices=["LR", "XGB", "RF", "NN"], required=True
    )
    
    parser.add_argument("--config", type=str, default=None, required=True)
    parser.add_argument("--lead_time", type=int, default=0, required=False)
    parser.add_argument("--use_dems", dest="use_dems", action="store_true")
    parser.add_argument("--n_inner_trials", type=int, default=250)
    parser.add_argument("--all_lead_times",default=True, dest="all_lead_times", action="store_true")
    parser.add_argument("--feature_set",type=str, default="all", required=False)
    args = parser.parse_args()

model = args.model
lead_time = args.lead_time
use_static = args.use_dems
all_lead_times = args.all_lead_times
n_inner_trials = args.n_inner_trials
feature_set = args.feature_set
config_file = args.config

project_dir, input_dir, output_dir, model_dir = config.get_dirs(config_file)
print('Project dir:', project_dir)
if all_lead_times:
    lead_times = [0, 30, 60, 90, 120, 150, 180, 270, 360, 450, 540, 630, 720]
    for lead_time in lead_times:
        data = get_features_at_lead_time(
            input_dir + f"\lead_time={lead_time}", use_static=True
        )
        X = data.drop(columns="OUTCOME")
        X = select_features(X, feature_set_name=feature_set)
        y = data["OUTCOME"]
        lead_time_results = nested_cross_val(
            model, X, y, lead_time, n_inner_trials=n_inner_trials, use_static=True, feature_set_name=feature_set,config_file=config_file
        )

else:
    data = get_features_at_lead_time(
        input_dir + f"\lead_time={lead_time}", use_static=True
    )
    X = data.drop(columns="OUTCOME")
    X = select_features(X, feature_set_name=feature_set)
    y = data["OUTCOME"]
    lead_time_results = nested_cross_val(
        "LR", X, y, lead_time, n_inner_trials=n_inner_trials, use_static=True, feature_set_name=feature_set, config_file=config_file
    )
