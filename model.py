"""Model klasyfikacji wyniku meczu (H/D/A) oparty o XGBoost."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import MODEL_PATH, XGB_PARAMS
from features import FEATURE_COLUMNS, add_league_features


RESULT_LABELS = ["H", "D", "A"]
RESULT_NAMES = {"H": "Wygrana gospodarzy", "D": "Remis", "A": "Wygrana gości"}


@dataclass
class FootballModel:
    """Pipeline + enkoder etykiet H/D/A."""

    pipeline: Pipeline
    label_encoder: LabelEncoder

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        encoded = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(encoded.astype(int))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    @property
    def classes_(self) -> list[str]:
        return list(self.label_encoder.classes_)


def _get_feature_columns() -> list[str]:
    return FEATURE_COLUMNS + ["league_code"]


def build_pipeline() -> Pipeline:
    numeric_features = _get_feature_columns()
    preprocessor = ColumnTransformer(
        transformers=[("num", "passthrough", numeric_features)],
        remainder="drop",
    )
    classifier = XGBClassifier(**XGB_PARAMS)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_model(
    training_df: pd.DataFrame,
    model_path: Path = MODEL_PATH,
    test_size: float = 0.2,
) -> tuple[FootballModel, dict]:
    """Trenuje model i zapisuje go na dysk."""
    if training_df.empty:
        raise ValueError("Brak danych treningowych.")

    data = add_league_features(training_df)
    X = data[_get_feature_columns()]
    y = data["ftr"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(RESULT_LABELS)
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    t0 = time.perf_counter()
    pipeline = build_pipeline()
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_enc)
    pipeline.fit(X_train, y_train_enc, classifier__sample_weight=sample_weight)

    model = FootballModel(pipeline=pipeline, label_encoder=label_encoder)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(
            y_test, y_pred, labels=RESULT_LABELS, zero_division=0
        ),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_seconds": round(time.perf_counter() - t0, 1),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model, metrics


def load_model(model_path: Path = MODEL_PATH) -> FootballModel:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model nie istnieje ({model_path}). Uruchom najpierw trening."
        )
    saved = joblib.load(model_path)
    if isinstance(saved, FootballModel):
        return saved
    # Kompatybilność ze starym modelem (Pipeline z regresją logistyczną)
    return saved


def predict_matches(
    model: FootballModel | Pipeline,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Zwraca prognozy z prawdopodobieństwami dla każdego wyniku."""
    if feature_df.empty:
        return pd.DataFrame()

    data = add_league_features(feature_df)
    X = data[_get_feature_columns()]

    if isinstance(model, FootballModel):
        proba = model.predict_proba(X)
        classes = model.classes_
        predicted = model.predict(X)
    else:
        proba = model.predict_proba(X)
        classes = list(model.named_steps["classifier"].classes_)
        predicted = model.predict(X)

    result = feature_df[
        ["div", "date", "home_team", "away_team"]
    ].copy()
    result["predicted"] = predicted

    for label in RESULT_LABELS:
        col = f"prob_{label}"
        result[col] = proba[:, classes.index(label)] if label in classes else np.nan

    result["confidence"] = proba.max(axis=1)
    result["prediction_label"] = result["predicted"].map(RESULT_NAMES)
    return result.sort_values(["div", "date"]).reset_index(drop=True)
