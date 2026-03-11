"""
Stacking Ensemble for Ligue 1 Predictor.
Base: LR + RF + XGBoost + MLP
Meta-learner: Calibrated LogisticRegression on cross_val_predict probabilities.
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

RANDOM_STATE = 42
LABEL_ORDER = ["H", "D", "A"]

META_COLS = [
    "fixture_id", "season", "date", "matchday",
    "home_team_id", "home_team", "away_team_id", "away_team",
    "referee", "result",
]


def load_data():
    """Load and split data."""
    df = pd.read_parquet(FEATURES_PATH)
    df = df[df["season"] >= 2022].copy()
    df = df.dropna(subset=["home_goals_avg"])

    feat_cols = [c for c in df.columns if c not in META_COLS]
    for col in feat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    train = df[df["season"].isin([2022, 2023])].copy()
    test = df[df["season"] == 2024].copy()

    le = LabelEncoder()
    le.classes_ = np.array(LABEL_ORDER)

    X_train = train[feat_cols].values.astype(np.float64)
    X_test = test[feat_cols].values.astype(np.float64)
    y_train = le.transform(train["result"].values)
    y_test = le.transform(test["result"].values)

    return X_train, X_test, y_train, y_test, feat_cols, le


def _xgb_sample_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    w = {c: n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([w[yi] for yi in y])


def build_base_models():
    """Return base models dict."""
    return {
        "lr": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5,
                C=1.0, class_weight="balanced", max_iter=5000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, eval_metric="mlogloss",
            verbosity=0, use_label_encoder=False,
        ),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation="relu",
                solver="adam", alpha=0.001, batch_size=32,
                learning_rate="adaptive", learning_rate_init=0.001,
                max_iter=500, early_stopping=True, validation_fraction=0.15,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def _brier_multi(y_true, y_proba, n_classes=3):
    y_onehot = np.eye(n_classes)[y_true]
    return np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))


def train_stacking():
    """Train stacking ensemble and evaluate."""
    X_train, X_test, y_train, y_test, feat_cols, le = load_data()
    tscv = TimeSeriesSplit(n_splits=5)

    print(f"Train: {len(y_train)} | Test: {len(y_test)} | Features: {len(feat_cols)}")

    base_models = build_base_models()

    # ── Step 1: Generate meta-features via cross_val_predict ──
    print("\nGenerating meta-features (cross_val_predict)...")
    meta_train_parts = []
    meta_test_parts = []

    for name, model in base_models.items():
        print(f"  {name}...", end=" ", flush=True)

        # Manual fold-based CV for all models (TimeSeriesSplit doesn't partition)
        meta_proba = np.full((len(y_train), 3), np.nan)

        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr = y_train[tr_idx]

            fold_model = clone(model)

            if name == "xgb":
                sw = _xgb_sample_weights(y_tr)
                fold_model.fit(X_tr, y_tr, sample_weight=sw)
            else:
                fold_model.fit(X_tr, y_tr)

            meta_proba[val_idx] = fold_model.predict_proba(X_val)

        # Fill rows not covered by any val fold with uniform proba
        nan_mask = np.isnan(meta_proba[:, 0])
        meta_proba[nan_mask] = 1.0 / 3.0

        # Refit on full training set
        if name == "xgb":
            sw_full = _xgb_sample_weights(y_train)
            model.fit(X_train, y_train, sample_weight=sw_full)
        else:
            model.fit(X_train, y_train)
        test_proba = model.predict_proba(X_test)

        meta_train_parts.append(meta_proba)
        meta_test_parts.append(test_proba)
        print(f"done (CV shape: {meta_proba.shape})")

    # Stack meta-features: (n_samples, n_models * 3)
    X_meta_train = np.hstack(meta_train_parts)
    X_meta_test = np.hstack(meta_test_parts)

    meta_col_names = []
    for name in base_models:
        for cls in LABEL_ORDER:
            meta_col_names.append(f"{name}_prob_{cls}")

    print(f"\nMeta-features shape: {X_meta_train.shape}")

    # ── Step 2: Train meta-learner ──
    print("Training meta-learner (Calibrated LR)...")
    meta_lr = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=3000,
        random_state=RANDOM_STATE, solver="lbfgs",
    )
    meta_model = CalibratedClassifierCV(meta_lr, cv=tscv, method="isotonic")
    meta_model.fit(X_meta_train, y_train)

    # ── Step 3: Evaluate ──
    y_pred = meta_model.predict(X_meta_test)
    y_proba = meta_model.predict_proba(X_meta_test)

    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    f1_per = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2])
    ll = log_loss(y_test, y_proba, labels=[0, 1, 2])
    roc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    brier = _brier_multi(y_test, y_proba)

    print(f"\n{'='*60}")
    print(f"STACKING ENSEMBLE RESULTS (holdout 2024)")
    print(f"{'='*60}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1 macro:    {f1_mac:.4f}")
    print(f"  F1 H/D/A:    {f1_per[0]:.4f} / {f1_per[1]:.4f} / {f1_per[2]:.4f}")
    print(f"  ROC AUC:     {roc:.4f}")
    print(f"  Log Loss:    {ll:.4f}")
    print(f"  Brier:       {brier:.4f}")
    print(f"{'='*60}")

    # Compare vs individual models
    comp_path = MODELS_DIR / "comparison.csv"
    if comp_path.exists():
        prev = pd.read_csv(comp_path)
        new_row = pd.DataFrame([{
            "model": "StackingEnsemble",
            "accuracy": acc, "f1_macro": f1_mac,
            "f1_H": f1_per[0], "f1_D": f1_per[1], "f1_A": f1_per[2],
            "roc_auc": roc, "log_loss": ll, "brier": brier,
        }])
        combined = pd.concat([prev[prev["model"] != "StackingEnsemble"], new_row], ignore_index=True)
        combined = combined.sort_values("f1_macro", ascending=False)
        combined.to_csv(comp_path, index=False)

        print(f"\nUPDATED COMPARISON TABLE:")
        pd.set_option("display.float_format", "{:.4f}".format)
        pd.set_option("display.width", 200)
        print(combined.to_string(index=False))

    # ── Step 4: Save everything ──
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save base models
    for name, model in base_models.items():
        joblib.dump(model, MODELS_DIR / f"stack_base_{name}.joblib")

    # Save meta-learner
    joblib.dump(meta_model, MODELS_DIR / f"stack_meta.joblib")

    # Save stacking config
    stack_config = {
        "base_model_names": list(base_models.keys()),
        "meta_col_names": meta_col_names,
        "feature_cols": feat_cols,
        "label_order": LABEL_ORDER,
    }
    joblib.dump(stack_config, MODELS_DIR / "stack_config.joblib")

    print(f"\nSaved stacking ensemble to {MODELS_DIR}")

    return meta_model, base_models, stack_config


def predict_stacking(X: np.ndarray) -> np.ndarray:
    """Load stacking ensemble and predict probabilities."""
    config = joblib.load(MODELS_DIR / "stack_config.joblib")
    meta_model = joblib.load(MODELS_DIR / "stack_meta.joblib")

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    meta_features = []
    for name in config["base_model_names"]:
        base = joblib.load(MODELS_DIR / f"stack_base_{name}.joblib")
        meta_features.append(base.predict_proba(X))

    X_meta = np.hstack(meta_features)
    return meta_model.predict_proba(X_meta)


if __name__ == "__main__":
    train_stacking()
