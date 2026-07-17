# Superconductivity Critical Temperature Prediction

This project predicts the critical temperature (`critical_temp`) of superconducting materials from statistical material features and chemical composition features. It was developed as a machine learning final project and is organized here as a reproducible GitHub project for portfolio and resume use.

## Highlights

- Built a regression pipeline for superconductivity critical temperature prediction.
- Added chemistry-aware feature engineering from material composition, including element fractions, element-group indicators, weighted atomic-property statistics, and domain-inspired compound-family features.
- Trained an ensemble of `ExtraTreesRegressor` and `LightGBMRegressor`.
- Reproduced validation results with a fixed 80/20 split.

## Result

| Metric | Validation score |
| --- | ---: |
| RMSE | 8.9479 |
| MAE | 4.9680 |
| R2 | 0.9303 |

Final validation ensemble weights:

| Model | Weight |
| --- | ---: |
| ExtraTreesRegressor | 0.30 |
| LightGBMRegressor | 0.70 |

## Project Structure

```text
.
├── data/
│   ├── ml_train.csv
│   └── unique_m_train.csv
├── docs/
│   └── final_report.pdf
├── src/
│   ├── chemistry_features.py
│   ├── train_feature_engineered.py
│   ├── validate_feature_engineered.py
│   └── predict_feature_engineered.py
├── requirements.txt
└── README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproduce Validation

```powershell
python src/validate_feature_engineered.py --train_data data/ml_train.csv --unique_data data/unique_m_train.csv
```

This trains the final model configuration on a fixed train/validation split and prints overall metrics plus metrics by temperature range.

## Train Final Model

```powershell
python src/train_feature_engineered.py --train_data data/ml_train.csv --unique_data data/unique_m_train.csv --model_path models/model_feature_engineered.joblib
```

The trained model file is intentionally not included in this repository because model artifacts can be large. It can be regenerated with the command above.

## Predict on New Data

Prepare row-aligned files:

- `ml_test.csv`: statistical features, optionally with `critical_temp` for evaluation
- `unique_m_test.csv`: element composition features

Then run:

```powershell
python src/predict_feature_engineered.py --test_data data/ml_test.csv --unique_data data/unique_m_test.csv --model_path models/model_feature_engineered.joblib --prediction_path predictions.csv
```

## Resume Summary

Portfolio wording:

> Superconductivity Critical Temperature Prediction: built a chemistry-aware regression pipeline using composition-derived features and an ExtraTrees + LightGBM ensemble, achieving RMSE 8.95 and R2 0.93 on a fixed validation split.
