"""
Ligue 1 Model Trainer
Trains 6 models, evaluates on 2024 holdout, saves results.
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

RANDOM_STATE = 42
LABEL_ORDER = ["H", "D", "A"]  # Home, Draw, Away

# Feature columns for the model (everything that's not metadata/target)
META_COLS = [
    "fixture_id", "season", "date", "matchday",
    "home_team_id", "home_team", "away_team_id", "away_team",
    "referee", "result",
]


def load_data():
    """Load features, filter 2022+, split train/test."""
    df = pd.read_parquet(FEATURES_PATH)

    # Filter to 2022+ only
    df = df[df["season"] >= 2022].copy()

    # Drop rows with no rolling history (first matchday)
    df = df.dropna(subset=["home_goals_avg"])

    # Feature columns
    feat_cols = [c for c in df.columns if c not in META_COLS]

    # Impute remaining NaN with column median
    for col in feat_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Split: 2022-2024 + first 70% of 2025 → train, last 30% of 2025 → test
    pre_2025 = df[df["season"].isin([2022, 2023, 2024])].copy()
    season_2025 = df[df["season"] == 2025].sort_values("date").copy()
    split_idx = int(len(season_2025) * 0.7)
    s2025_train = season_2025.iloc[:split_idx]
    s2025_test = season_2025.iloc[split_idx:]

    train = pd.concat([pre_2025, s2025_train])
    test = s2025_test.copy()

    X_train = train[feat_cols].values
    X_test = test[feat_cols].values

    le = LabelEncoder()
    le.classes_ = np.array(LABEL_ORDER)
    y_train = le.transform(train["result"].values)
    y_test = le.transform(test["result"].values)

    print(f"Features: {len(feat_cols)}")
    print(f"Train: {len(train)} matches (2022-2024 + 70% of 2025)")
    print(f"  Last train match: {train['date'].max()}  {train.iloc[-1]['home_team']} vs {train.iloc[-1]['away_team']}")
    print(f"Test:  {len(test)} matches (last 30% of 2025)")
    print(f"  First test match: {test['date'].min()}  {test.iloc[0]['home_team']} vs {test.iloc[0]['away_team']}")
    print(f"  Last test match:  {test['date'].max()}  {test.iloc[-1]['home_team']} vs {test.iloc[-1]['away_team']}")
    print(f"Train target dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test  target dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    return X_train, X_test, y_train, y_test, feat_cols, le


def _brier_multi(y_true, y_proba, n_classes=3):
    """Multiclass Brier score (lower is better)."""
    y_onehot = np.eye(n_classes)[y_true]
    return np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))


def evaluate(name, y_true, y_pred, y_proba, le):
    """Compute all metrics for a model."""
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro")
    f1_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    ll = log_loss(y_true, y_proba, labels=[0, 1, 2])
    brier = _brier_multi(y_true, y_proba)

    try:
        roc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        roc = np.nan

    return {
        "model": name,
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_H": f1_per[0],
        "f1_D": f1_per[1],
        "f1_A": f1_per[2],
        "roc_auc": roc,
        "log_loss": ll,
        "brier": brier,
    }


def build_models():
    """Return dict of model_name → model."""
    tscv = TimeSeriesSplit(n_splits=5)

    models = {}

    # 1. Logistic Regression (elasticnet)
    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )),
    ])

    # 2. Random Forest + Calibration
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    models["RandomForest"] = CalibratedClassifierCV(
        rf, cv=tscv, method="isotonic",
    )

    # 3. XGBoost
    # Compute sample weights for balanced classes
    models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        verbosity=0,
        use_label_encoder=False,
    )

    # 4. LightGBM
    models["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    # 5. SVM + Calibration
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced",
            probability=False,
            random_state=RANDOM_STATE,
        )),
    ])
    models["SVM"] = CalibratedClassifierCV(
        svm, cv=tscv, method="sigmoid",
    )

    # 6. MLP Neural Network (sklearn)
    models["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=32,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        )),
    ])

    return models


def _compute_xgb_sample_weights(y_train):
    """Compute sample weights to simulate class_weight='balanced' for XGBoost."""
    classes, counts = np.unique(y_train, return_counts=True)
    n_samples = len(y_train)
    n_classes = len(classes)
    weights = n_samples / (n_classes * counts)
    weight_map = dict(zip(classes, weights))
    return np.array([weight_map[y] for y in y_train])


def train_and_evaluate():
    """Main training loop."""
    X_train, X_test, y_train, y_test, feat_cols, le = load_data()

    models = build_models()
    results = []

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        if name == "XGBoost":
            sample_weights = _compute_xgb_sample_weights(y_train)
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Evaluate
        metrics = evaluate(name, y_test, y_pred, y_proba, le)
        results.append(metrics)

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 macro: {metrics['f1_macro']:.4f}")
        print(f"  Log Loss: {metrics['log_loss']:.4f}")

        # Save model
        model_path = MODELS_DIR / f"{name.lower()}.joblib"
        joblib.dump(model, model_path)
        print(f"  Saved → {model_path.name}")

    # Save label encoder and feature names
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    joblib.dump(feat_cols, MODELS_DIR / "feature_cols.joblib")

    # Results table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("f1_macro", ascending=False)

    print(f"\n{'='*70}")
    print("COMPARISON TABLE (sorted by F1 macro)")
    print(f"{'='*70}")

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(results_df.to_string(index=False))

    # Winner
    best = results_df.iloc[0]
    print(f"\n{'='*70}")
    print(f"WINNER: {best['model']}")
    print(f"  Accuracy:  {best['accuracy']:.4f}")
    print(f"  F1 macro:  {best['f1_macro']:.4f}")
    print(f"  F1 H/D/A:  {best['f1_H']:.4f} / {best['f1_D']:.4f} / {best['f1_A']:.4f}")
    print(f"  ROC AUC:   {best['roc_auc']:.4f}")
    print(f"  Log Loss:  {best['log_loss']:.4f}")
    print(f"  Brier:     {best['brier']:.4f}")
    print(f"{'='*70}")

    # Save results
    results_df.to_csv(MODELS_DIR / "comparison.csv", index=False)

    return results_df


if __name__ == "__main__":
    train_and_evaluate()
