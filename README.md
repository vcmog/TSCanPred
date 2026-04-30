# TS_CANPRED

Time-series prediction framework for training and evaluating deep learning and traditional ML models on sequential data.

## Quick Start

The main entry point is:

```bash
python run_DL_cross_val.py
```

## Example

```bash
python run_DL_cross_val.py \
  --model LSTM \
  --config configs/config.yaml \
  --lead_time 7
```

## Required Arguments
- model
Model to train: LSTM, GRU, CNN, Transformer, LSTM_ALT
- config
Path to config file (defines data and output directories)
- lead_time
Prediction horizon (e.g. 7 days)
## Common Optional Arguments
- model_type (default: RNN-lasthiddenstate)
- n_epochs (default: 10)
- use_dems (include static features)
- hyperparameter_tune (enable tuning)
- start_fold (resume cross-validation)
## What It Does
- Prepares data (if not already processed)
- Runs k-fold cross-validation
- Trains model(s)
- Evaluates performance
- Saves results and predictions

## Outputs

Results are saved to the directory specified in your config file.

## Project Structure (brief)
- run_DL_cross_val.py → main training pipeline
- models/ → model definitions
- training/ → training logic
- datasets/ → data handling
- features/ → feature engineering
- evaluation/ → metrics

## Notes
Data must be configured via the config file
GRU-D has a separate script: run_GRUD_cross-val.py
Sklearn models use: run_sklearn_cross_val.py