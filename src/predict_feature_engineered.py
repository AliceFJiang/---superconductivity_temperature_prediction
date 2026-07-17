from __future__ import annotations

import argparse
import os
import pickle
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

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from chemistry_features import build_composition_features


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

TARGET = "critical_temp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate feature-engineered task 3 model.")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--unique_data", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prediction_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.model_path).open("rb") as f:
        bundle = pickle.load(f)

    test_df = pd.read_csv(args.test_data)
    unique_df = pd.read_csv(args.unique_data)
    if len(test_df) != len(unique_df):
        raise ValueError("test_data and unique_data must have the same number of rows.")

    base_x = test_df[bundle["base_feature_columns"]]
    chem_x = build_composition_features(unique_df)
    x = pd.concat([base_x, chem_x], axis=1)
    x = x[bundle["feature_columns"]]

    weights = bundle["ensemble_weights"]
    pred = (
        weights["extra_trees"] * bundle["models"]["extra_trees"].predict(x)
        + weights["lightgbm"] * bundle["models"]["lightgbm"].predict(x)
    )

    print(f"test shape: {test_df.shape}, unique shape: {unique_df.shape}")
    print(
        "feature-engineered ensemble weights: "
        f"ExtraTrees={weights['extra_trees']:.2f}, "
        f"LightGBM={weights['lightgbm']:.2f}"
    )
    if TARGET in test_df.columns:
        y_true = test_df[TARGET]
        print(f"RMSE: {mean_squared_error(y_true, pred) ** 0.5:.4f}")
        print(f"MAE: {mean_absolute_error(y_true, pred):.4f}")
        print(f"R2: {r2_score(y_true, pred):.4f}")
    else:
        print(f"predicted samples: {len(pred)}")

    if args.prediction_path:
        pd.DataFrame({"prediction": pred}).to_csv(args.prediction_path, index=False)
        print(f"saved predictions: {args.prediction_path}")


if __name__ == "__main__":
    main()
