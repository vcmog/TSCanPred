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

- `--model`  
  Model to train: LSTM, GRU, CNN, Transformer, LSTM_ALT  

- `--config`  
  Path to config file (defines data and output directories)  

- `--lead_time`  
  Prediction horizon (e.g. 7 days)

## Common Optional Arguments

- `--model_type` (default: RNN-lasthiddenstate)  
- `--n_epochs` (default: 10)  
- `--use_dems` (include static features)  
- `--hyperparameter_tune` (enable tuning)  
- `--start_fold` (resume cross-validation)

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
- Data must be configured via the config file
- GRU-D has a separate script: run_GRUD_cross-val.py
- Sklearn models use: run_sklearn_cross_val.py





## Sklearn Cross-Validation Pipeline

This script runs nested cross-validation for classical machine learning models
(Logistic Regression, Random Forest, XGBoost, and Neural Networks) across one or
multiple prediction lead times.

---

## Key Imports / Components

- `get_features_at_lead_time` → loads feature dataset for a specific forecast horizon  
- `select_features` → applies feature subset selection  
- `nested_cross_val` → performs nested cross-validation (inner hyperparameter tuning + outer evaluation)  
- `config.get_dirs` → resolves project directory structure  

---

## Command-Line Arguments

### Required

- `--model`  
  Model type: `LR`, `XGB`, `RF`, `NN`

- `--config`  
  Path to configuration file defining data/project structure  

### Optional

- `--lead_time` (default: 0)  
  Forecast horizon in days  

- `--use_dems`  
  Include static (DEM) features  

- `--n_inner_trials` (default: 250)  
  Number of hyperparameter optimisation trials in inner CV loop  

- `--all_lead_times` (default: True)  
  Run evaluation across multiple predefined lead times  

- `--feature_set` (default: "all")  
  Select subset of features  

---

## Feature Engineering (Important)

Features are based on **trend differences between two time windows** for each blood test variable:

- **Proximal window** → recent measurements closer to prediction time  
- **Distal window** → earlier baseline measurements further in the past  

For each feature, the pipeline constructs:

- Change between proximal and distal windows  
- Relative change (trend direction and magnitude)  
- Window-based summary statistics (e.g., mean/variance depending on feature)

### Purpose

This allows the model to learn **temporal trends in patient physiology**, rather than relying only on static values.

---

## Workflow

1. Parse command-line arguments  
2. Load directories from config file  
3. For each lead time (or single lead time):
   - Load precomputed feature dataset via `get_features_at_lead_time`
   - Extract target variable `OUTCOME`
   - Apply feature selection via `select_features`
   - Run `nested_cross_val` for training + evaluation  

---

## Lead Time Strategy

If `--all_lead_times` is enabled, training is repeated for:

```python
[0, 30, 60, 90, 120, 150, 180, 270, 360, 450, 540, 630, 720]
```

Each lead time:
- Uses its own dataset folder  
- Runs independent nested cross-validation  
- Produces separate evaluation results  

---

## Output

For each run, the pipeline outputs:

- Cross-validation performance metrics  
- Hyperparameter tuning results  
- Lead-time-specific evaluation results  

Saved to the output directory defined in the config file.