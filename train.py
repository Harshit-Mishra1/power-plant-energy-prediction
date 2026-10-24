"""Train and evaluate a PyTorch ANN for power plant energy output prediction."""
from pathlib import Path
import json
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "powerplant_data.csv"
ARTIFACTS = ROOT / "artifacts"
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


class PowerPlantANN(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    set_seed()
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"AT", "V", "AP", "RH", "PE"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")
    if df[list(required)].isnull().any().any():
        raise ValueError("Dataset contains missing values; clean them before training.")

    # AT=ambient temperature, V=exhaust vacuum, AP=ambient pressure,
    # RH=relative humidity, PE=net hourly electrical energy output.
    X = df[["AT", "V", "AP", "RH"]].astype("float32")
    y = df["PE"].astype("float32")

    # Split once into train/validation/test. The test set is held out until final evaluation.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, random_state=SEED
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype("float32")
    X_val_s = scaler.transform(X_val).astype("float32")
    X_test_s = scaler.transform(X_test).astype("float32")

    def tensor_dataset(x, target):
        return TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(target.to_numpy().reshape(-1, 1), dtype=torch.float32),
        )

    train_loader = DataLoader(tensor_dataset(X_train_s, y_train), batch_size=64, shuffle=True)
    val_ds = tensor_dataset(X_val_s, y_val)
    X_val_t = val_ds.tensors[0]
    y_val_t = val_ds.tensors[1]

    model = PowerPlantANN(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val_loss = float("inf")
    best_state = None
    patience, wait, epochs = 20, 0, 150
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch + 1:03d} | train MSE={train_loss:.4f} | val MSE={val_loss:.4f}")
        if wait >= patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on the held-out test set only.
    model.eval()
    with torch.no_grad():
        ann_pred = model(torch.tensor(X_test_s, dtype=torch.float32)).numpy().ravel()
    ann_metrics = metrics(y_test, ann_pred)

    # Baseline comparison: fit preprocessing only on training data.
    baseline = LinearRegression().fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = metrics(y_test, baseline_pred)

    ARTIFACTS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ARTIFACTS / "ann_model_state.pt")
    joblib.dump(scaler, ARTIFACTS / "feature_scaler.joblib")
    results = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "ann_test_metrics": ann_metrics,
        "linear_regression_test_metrics": baseline_metrics,
        "best_validation_mse": float(best_val_loss),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame({"actual_PE": y_test.to_numpy(), "ann_predicted_PE": ann_pred,
                  "linear_regression_predicted_PE": baseline_pred}).to_csv(ARTIFACTS / "test_predictions.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training MSE")
    plt.plot(val_losses, label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("ANN Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "training_validation_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, ann_pred, alpha=0.5)
    lo = float(min(y_test.min(), ann_pred.min()))
    hi = float(max(y_test.max(), ann_pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual PE")
    plt.ylabel("Predicted PE")
    plt.title("ANN: Actual vs Predicted on Test Set")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "actual_vs_predicted.png", dpi=160)
    plt.close()

    print("\nFinal held-out test results")
    print("ANN:", json.dumps(ann_metrics, indent=2))
    print("Linear Regression baseline:", json.dumps(baseline_metrics, indent=2))
    print(f"\nSaved outputs to: {ARTIFACTS}")


if __name__ == "__main__":
    main()
