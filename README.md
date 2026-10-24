# Power Plant Energy Output Prediction with PyTorch ANN

A supervised regression project that predicts net electrical energy output (`PE`) from ambient temperature (`AT`), exhaust vacuum (`V`), ambient pressure (`AP`), and relative humidity (`RH`) using a feedforward Artificial Neural Network built with PyTorch.

## Project highlights

- Data checks and a reproducible train/validation/test split
- Feature standardization fitted on the training split only
- Feedforward ANN trained with mini-batches, Adam, MSE loss, and early stopping
- Held-out test evaluation with MAE, RMSE, and R²
- Linear Regression baseline for comparison
- Training/validation loss plot and actual-versus-predicted plot
- Saved model weights, scaler, metrics, and test predictions (generated locally)

## Dataset

`powerplant_data.csv` contains 9,568 rows and five columns. The target is `PE`; input features are `AT`, `V`, `AP`, and `RH`. The dataset is included in this repository for reproducibility. Confirm the dataset's license and attribution requirements before redistribution if you obtained it from a third-party source.

## Setup (Windows / VS Code)

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
```

On macOS/Linux, activate the environment with `source .venv/bin/activate` instead.

## Outputs

After training, the script creates an `artifacts/` directory containing:

- `ann_model_state.pt` — trained ANN weights
- `feature_scaler.joblib` — fitted feature scaler
- `metrics.json` — split sizes and test metrics
- `test_predictions.csv` — actual and predicted test values
- `training_validation_loss.png` — learning curves
- `actual_vs_predicted.png` — prediction diagnostic plot

The generated `artifacts/` folder is ignored by Git by default; run `python train.py` to recreate it. Do not report performance numbers in a resume until you have run the code and confirmed the metrics in `artifacts/metrics.json`.

## Methodology

1. Load the CSV and validate required columns / missing values.
2. Split the data into training, validation, and held-out test sets (64% / 16% / 20%).
3. Fit `StandardScaler` on the training inputs only and apply it to validation/test inputs.
4. Train a multilayer perceptron with ReLU hidden activations and a single continuous output.
5. Select the checkpoint with the best validation MSE and stop early when validation performance stops improving.
6. Evaluate once on the held-out test set and compare against Linear Regression.

## Limitations

This is an educational tabular regression project, not a deployed industrial control system. Results depend on the dataset and split. A single random split does not establish generalization to new plants or future operating conditions.

## Tech stack

Python · PyTorch · Pandas · NumPy · Scikit-learn · Matplotlib
