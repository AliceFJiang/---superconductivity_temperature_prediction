from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train feature-engineered task 3 model.")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--unique_data", required=True)
    parser.add_argument("--model_path", default="model_feature_engineered.joblib")
    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--et_estimators", type=int, default=600)
    parser.add_argument("--et_max_features", type=float, default=0.7)
    parser.add_argument("--lgbm_estimators", type=int, default=1200)
    parser.add_argument("--lgbm_learning_rate", type=float, default=0.03)
    parser.add_argument("--lgbm_num_leaves", type=int, default=31)
    parser.add_argument("--lgbm_min_child_samples", type=int, default=20)
    parser.add_argument("--lgbm_colsample_bytree", type=float, default=0.9)
    parser.add_argument("--lgbm_subsample", type=float, default=0.9)
    parser.add_argument("--lgbm_reg_alpha", type=float, default=0.05)
    parser.add_argument("--lgbm_reg_lambda", type=float, default=0.1)
    parser.add_argument("--high_temp_threshold", type=float, default=77.0)
    parser.add_argument(
        "--high_temp_weight",
        type=float,
        default=1.0,
        help="Training weight for high-temperature samples. Use 1.0 to disable weighting.",
    )
    parser.add_argument("--n_jobs", type=int, default=None)
    return parser.parse_args()


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": mean_squared_error(y_true, pred) ** 0.5,
        "mae": mean_absolute_error(y_true, pred),
        "r2": r2_score(y_true, pred),
    }


def best_weight(y_true: np.ndarray, et_pred: np.ndarray, lgbm_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    best = 0.5
    best_metrics: dict[str, float] | None = None
    for weight in np.linspace(0.0, 1.0, 21):
        pred = weight * et_pred + (1.0 - weight) * lgbm_pred
        current = metrics(y_true, pred)
        if best_metrics is None or current["rmse"] < best_metrics["rmse"]:
            best = float(weight)
            best_metrics = current
    assert best_metrics is not None
    return best, best_metrics


def make_extra_trees(args: argparse.Namespace) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=args.et_estimators,
        max_features=args.et_max_features,
        min_samples_leaf=1,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )


def make_lightgbm(args: argparse.Namespace) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=args.lgbm_estimators,
        learning_rate=args.lgbm_learning_rate,
        num_leaves=args.lgbm_num_leaves,
        max_depth=-1,
        min_child_samples=args.lgbm_min_child_samples,
        subsample=args.lgbm_subsample,
        colsample_bytree=args.lgbm_colsample_bytree,
        reg_alpha=args.lgbm_reg_alpha,
        reg_lambda=args.lgbm_reg_lambda,
        objective="regression",
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=-1,
    )


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


def make_sample_weight(args: argparse.Namespace, y: pd.Series) -> np.ndarray:
    weights = np.ones(len(y), dtype=float)
    weights[y.to_numpy() >= args.high_temp_threshold] = args.high_temp_weight
    return weights


def train_pair(args: argparse.Namespace, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    sample_weight = make_sample_weight(args, y)
    extra_trees = make_extra_trees(args)
    lightgbm = make_lightgbm(args)
    extra_trees.fit(x, y, sample_weight=sample_weight)
    lightgbm.fit(x, y, sample_weight=sample_weight)
    return {
        "extra_trees": extra_trees,
        "lightgbm": lightgbm,
    }


def main() -> None:
    args = parse_args()
    x, y = load_features(Path(args.train_data), Path(args.unique_data))
    x_fit, x_valid, y_fit, y_valid = train_test_split(
        x, y, test_size=args.valid_size, random_state=args.random_state
    )

    valid_models = train_pair(args, x_fit, y_fit)
    et_pred = valid_models["extra_trees"].predict(x_valid)
    lgbm_pred = valid_models["lightgbm"].predict(x_valid)
    et_metrics = metrics(y_valid.to_numpy(), et_pred)
    lgbm_metrics = metrics(y_valid.to_numpy(), lgbm_pred)
    et_weight, ensemble_metrics = best_weight(y_valid.to_numpy(), et_pred, lgbm_pred)

    print(f"feature shape: {x.shape}")
    print(
        "high-temperature weighting: "
        f"threshold={args.high_temp_threshold:.1f}K, "
        f"weight={args.high_temp_weight:.2f}"
    )
    print(f"ExtraTrees: RMSE={et_metrics['rmse']:.4f}, MAE={et_metrics['mae']:.4f}, R2={et_metrics['r2']:.4f}")
    print(f"LightGBM: RMSE={lgbm_metrics['rmse']:.4f}, MAE={lgbm_metrics['mae']:.4f}, R2={lgbm_metrics['r2']:.4f}")
    print(
        f"Feature ensemble (ExtraTrees weight={et_weight:.2f}): "
        f"RMSE={ensemble_metrics['rmse']:.4f}, "
        f"MAE={ensemble_metrics['mae']:.4f}, "
        f"R2={ensemble_metrics['r2']:.4f}"
    )

    final_models = train_pair(args, x, y)
    bundle = {
        "model_type": "feature_engineered_ensemble",
        "target": TARGET,
        "feature_columns": list(x.columns),
        "base_feature_columns": [col for col in pd.read_csv(args.train_data, nrows=1).columns if col != TARGET],
        "high_temp_threshold": args.high_temp_threshold,
        "high_temp_weight": args.high_temp_weight,
        "ensemble_weights": {
            "extra_trees": et_weight,
            "lightgbm": 1.0 - et_weight,
        },
        "models": final_models,
        "validation_results": {
            "extra_trees": et_metrics,
            "lightgbm": lgbm_metrics,
            "ensemble": ensemble_metrics,
        },
    }
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"saved feature-engineered model: {model_path}")


if __name__ == "__main__":
    main()
