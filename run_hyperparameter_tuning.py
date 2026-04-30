import argparse
import hyperparameter_tune.hyperparameter_tune as tune

if __name__ == "__main__":
    # initialise argument parser
    parser = argparse.ArgumentParser()

    # add arguments to parse
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()

if args.model == "LSTM":
    tune.tune_lstm(n_trials=args.n_trials)

if args.model == "GRU":
    tune.tune_gru(n_trials=args.n_trials)

if args.model == "LSTM_ALT":
    tune.tune_lstm_alt(n_trials=args.n_trials)

if args.model == "CNN":
    tune.tune_cnn(n_trials=args.n_trials)

if args.model == "LR":
    "Hyperparameter tuning logistic regression model"
    tune.hyperparameter_tune_LR(n_trials=args.n_trials)
