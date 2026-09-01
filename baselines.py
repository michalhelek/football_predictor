"""Benchmark kursów bukmacherskich i ensemble Dixon-Coles + XGBoost."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss

from dixon_coles import RESULT_LABELS, RESULT_NAMES
from features import _odds_features

KEY_COLUMNS = ["div", "date", "home_team", "away_team"]


def implied_probs(avg_h, avg_d, avg_a) -> dict[str, float] | None:
    """Prawdopodobieństwa implikowane z kursów (z normalizacją marży)."""
    odds = _odds_features(avg_h, avg_d, avg_a)
    h, d, a = odds["implied_prob_h"], odds["implied_prob_d"], odds["implied_prob_a"]
    if any(pd.isna(x) for x in (h, d, a)):
        return None
    return {"H": float(h), "D": float(d), "A": float(a)}


def predict_matches_odds(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Prognoza H/D/A wyłącznie z kursów AvgH/D/A (benchmark rynku)."""
    if fixtures.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for row in fixtures.itertuples():
        probs = implied_probs(
            getattr(row, "avg_h", None),
            getattr(row, "avg_d", None),
            getattr(row, "avg_a", None),
        )
        if probs is None:
            continue

        predicted = max(probs, key=probs.get)
        rows.append(
            {
                "div": row.div,
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "predicted": predicted,
                "prob_H": probs["H"],
                "prob_D": probs["D"],
                "prob_A": probs["A"],
                "confidence": max(probs.values()),
                "prediction_label": RESULT_NAMES[predicted],
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["div", "date"]).reset_index(drop=True)


def ensemble_predictions(
    *prediction_frames: pd.DataFrame,
    weights: list[float] | None = None,
) -> pd.DataFrame:
    """Łączy prognozy przez ważoną średnią prawdopodobieństw H/D/A."""
    frames = [df for df in prediction_frames if not df.empty]
    if not frames:
        return pd.DataFrame()

    if weights is None:
        weights = [1.0 / len(frames)] * len(frames)
    if len(weights) != len(frames):
        raise ValueError("Liczba wag musi odpowiadać liczbie modeli.")

    merged = frames[0][KEY_COLUMNS + ["prob_H", "prob_D", "prob_A"]].copy()
    merged["date"] = pd.to_datetime(merged["date"])
    merged["prob_H"] = merged["prob_H"] * weights[0]
    merged["prob_D"] = merged["prob_D"] * weights[0]
    merged["prob_A"] = merged["prob_A"] * weights[0]

    for weight, frame in zip(weights[1:], frames[1:], strict=False):
        other = frame[KEY_COLUMNS + ["prob_H", "prob_D", "prob_A"]].copy()
        other["date"] = pd.to_datetime(other["date"])
        merged = merged.merge(other, on=KEY_COLUMNS, how="inner", suffixes=("", "_other"))
        if merged.empty:
            return pd.DataFrame()
        merged["prob_H"] = merged["prob_H"] + merged["prob_H_other"] * weight
        merged["prob_D"] = merged["prob_D"] + merged["prob_D_other"] * weight
        merged["prob_A"] = merged["prob_A"] + merged["prob_A_other"] * weight
        merged = merged.drop(columns=["prob_H_other", "prob_D_other", "prob_A_other"])

    prob_cols = merged[["prob_H", "prob_D", "prob_A"]].values
    best_idx = prob_cols.argmax(axis=1)
    label_map = {0: "H", 1: "D", 2: "A"}
    merged["predicted"] = [label_map[i] for i in best_idx]
    merged["confidence"] = prob_cols.max(axis=1)
    merged["prediction_label"] = merged["predicted"].map(RESULT_NAMES)
    return merged.sort_values(["div", "date"]).reset_index(drop=True)


def evaluate_predictions(
    predictions: pd.DataFrame,
    test_matches: pd.DataFrame,
) -> dict:
    """Ocena gotowych prognoz na zbiorze testowym z znanym FTR."""
    if predictions.empty:
        return {"accuracy": 0.0, "log_loss": None, "samples": 0, "report": ""}

    test = test_matches.copy()
    test["date"] = pd.to_datetime(test["date"])
    preds = predictions.copy()
    preds["date"] = pd.to_datetime(preds["date"])

    merged = test.merge(
        preds[KEY_COLUMNS + ["predicted", "prob_H", "prob_D", "prob_A"]],
        on=KEY_COLUMNS,
        how="inner",
    )
    if merged.empty:
        return {"accuracy": 0.0, "log_loss": None, "samples": 0, "report": ""}

    y_true = merged["ftr"]
    y_pred = merged["predicted"]
    proba = merged[["prob_A", "prob_D", "prob_H"]].values

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, proba, labels=["A", "D", "H"])),
        "samples": len(merged),
        "report": classification_report(
            y_true, y_pred, labels=RESULT_LABELS, zero_division=0
        ),
    }
