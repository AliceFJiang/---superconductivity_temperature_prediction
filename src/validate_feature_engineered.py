from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".python_deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from chemistry_features import build_composition_features


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

TARGET = "critical_temp"
TEMP_BINS = [-np.inf, 10.0, 30.0, 77.0, np.inf]
TEMP_LABELS = ["<10K", "10-30K", "30-77K", ">=77K"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the final feature-engineered model on a fixed 80/20 split."
    )
    parser.add_argument("--train_data", required=True, help="Path to ml_train.csv.")
    parser.add_argument("--unique_data", required=True, help="Path to unique_m_train.csv.")
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    return parser.parse_args()


def load_features(train_path: Path, unique_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    train_df = pd.read_csv(train_path)
    unique_df = pd.read_csv(unique_path)
    if len(train_df) != len(unique_df):
        raise ValueError("train_data and unique_data must have the same number of rows.")
    if TARGET in unique_df.columns and not np.allclose(train_df[TARGET], unique_df[TARGET]):
        raise ValueError("Target columns are not aligned between train_data and unique_data.")

    base_x = train_df.drop(columns=[TARGET])
    chem_x = build_composition_features(unique_df)
    return pd.concat([base_x, chem_x], axis=1), train_df[TARGET]


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - y_true
    return {
        "rmse": mean_squared_error(y_true, pred) ** 0.5,
        "mae": mean_absolute_error(y_true, pred),
        "r2": r2_score(y_true, pred),
        "mean_residual": float(np.mean(residual)),
    }


def print_metrics(name: str, y_true: np.ndarray, pred: np.ndarray) -> None:
    current = metrics(y_true, pred)
    print(
        f"{name}: RMSE={current['rmse']:.4f}, "
        f"MAE={current['mae']:.4f}, "
        f"R2={current['r2']:.4f}, "
        f"mean_residual={current['mean_residual']:.4f}"
    )


def best_weight(y_true: np.ndarray, et_pred: np.ndarray, lgbm_pred: np.ndarray) -> tuple[float, np.ndarray]:
    best_w = 0.5
    best_pred = 0.5 * et_pred + 0.5 * lgbm_pred
    best_rmse = np.inf
    for w in np.linspace(0.0, 1.0, 21):
        pred = w * et_pred + (1.0 - w) * lgbm_pred
        rmse = mean_squared_error(y_true, pred) ** 0.5
        if rmse < best_rmse:
            best_w = float(w)
            best_pred = pred
            best_rmse = float(rmse)
    return best_w, best_pred


def main() -> None:
    args = parse_args()
    x, y = load_features(Path(args.train_data), Path(args.unique_data))
    x_fit, x_valid, y_fit, y_valid = train_test_split(
        x,
        y,
        test_size=args.valid_size,
        random_state=args.random_state,
    )
    y_valid_np = y_valid.to_numpy()

    extra_trees = ExtraTreesRegressor(
        n_estimators=600,
        max_features=0.7,
        min_samples_leaf=1,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    lightgbm = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.05,
        objective="regression",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=-1,
    )

    extra_trees.fit(x_fit, y_fit)
    lightgbm.fit(x_fit, y_fit)
    et_pred = extra_trees.predict(x_valid)
    lgbm_pred = lightgbm.predict(x_valid)
    et_weight, ensemble_pred = best_weight(y_valid_np, et_pred, lgbm_pred)

    print(f"feature shape: {x.shape}")
    print(f"train shape: {x_fit.shape}, valid shape: {x_valid.shape}")
    print_metrics("ExtraTreesRegressor", y_valid_np, et_pred)
    print_metrics("LightGBMRegressor", y_valid_np, lgbm_pred)
    print(
        "Final ensemble weights: "
        f"ExtraTrees={et_weight:.2f}, LightGBM={1.0 - et_weight:.2f}"
    )
    print_metrics("Final tuned ensemble", y_valid_np, ensemble_pred)

    result = pd.DataFrame({"y_true": y_valid_np, "y_pred": ensemble_pred})
    result["temp_group"] = pd.cut(
        result["y_true"], TEMP_BINS, labels=TEMP_LABELS, right=False
    )
    print("-- metrics by true temperature group --")
    for label in TEMP_LABELS:
        part = result[result["temp_group"] == label]
        print_metrics(str(label), part["y_true"].to_numpy(), part["y_pred"].to_numpy())


if __name__ == "__main__":
    main()
