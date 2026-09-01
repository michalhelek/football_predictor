"""Archiwizacja prognoz i ocena trafności po rozegraniu meczów."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from config import PREDICTION_HISTORY_PATH


HISTORY_COLUMNS = [
    "saved_at",
    "model",
    "league",
    "div",
    "date",
    "home_team",
    "away_team",
    "predicted",
    "prediction_label",
    "prob_H",
    "prob_D",
    "prob_A",
    "confidence",
]


def archive_predictions(predictions: pd.DataFrame) -> None:
    """Dopisuje prognozy do archiwum (nadpisuje ten sam mecz+model)."""
    if predictions.empty:
        return

    PREDICTION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    archive = predictions.copy()
    archive["saved_at"] = datetime.now().isoformat()
    archive["date"] = pd.to_datetime(archive["date"]).dt.strftime("%Y-%m-%d")

    for col in HISTORY_COLUMNS:
        if col not in archive.columns:
            archive[col] = None
    archive = archive[HISTORY_COLUMNS]

    if PREDICTION_HISTORY_PATH.exists():
        existing = pd.read_csv(PREDICTION_HISTORY_PATH, encoding="utf-8-sig")
        combined = pd.concat([existing, archive], ignore_index=True)
    else:
        combined = archive

    combined = combined.drop_duplicates(
        subset=["div", "date", "home_team", "away_team", "model"],
        keep="last",
    )
    combined = combined.sort_values(["date", "div"]).reset_index(drop=True)
    combined.to_csv(PREDICTION_HISTORY_PATH, index=False, encoding="utf-8-sig")


def evaluate_history(
    history: pd.DataFrame,
    played: pd.DataFrame,
) -> pd.DataFrame:
    """Łączy archiwum prognoz z rzeczywistymi wynikami z bazy."""
    if history.empty or played.empty:
        return history

    df = history.copy()
    df["date"] = pd.to_datetime(df["date"])
    played = played.copy()
    played["date"] = pd.to_datetime(played["date"])

    results = played[
        ["div", "date", "home_team", "away_team", "fthg", "ftag", "ftr"]
    ].rename(
        columns={
            "fthg": "actual_home_goals",
            "ftag": "actual_away_goals",
            "ftr": "actual_ftr",
        }
    )

    merged = df.merge(
        results,
        on=["div", "date", "home_team", "away_team"],
        how="left",
    )
    merged["evaluated"] = merged["actual_ftr"].isin(["H", "D", "A"])
    merged["correct"] = merged["evaluated"] & (merged["predicted"] == merged["actual_ftr"])
    return merged


def load_prediction_history(played: pd.DataFrame | None = None) -> pd.DataFrame:
    """Wczytuje archiwum prognoz z opcjonalną oceną trafności."""
    if not PREDICTION_HISTORY_PATH.exists():
        return pd.DataFrame()

    history = pd.read_csv(PREDICTION_HISTORY_PATH, encoding="utf-8-sig")
    if history.empty:
        return history

    if played is not None:
        return evaluate_history(history, played)

    history["date"] = pd.to_datetime(history["date"])
    return history


def history_summary(evaluated: pd.DataFrame) -> dict:
    """Zbiorcze metryki trafności prognoz."""
    if evaluated.empty or "evaluated" not in evaluated.columns:
        return {
            "total_predictions": len(evaluated),
            "evaluated": 0,
            "correct": 0,
            "accuracy": None,
            "by_model": pd.DataFrame(),
            "by_league": pd.DataFrame(),
        }

    completed = evaluated[evaluated["evaluated"]].copy()
    if completed.empty:
        return {
            "total_predictions": len(evaluated),
            "evaluated": 0,
            "correct": 0,
            "accuracy": None,
            "by_model": pd.DataFrame(),
            "by_league": pd.DataFrame(),
        }

    by_model = (
        completed.groupby("model")["correct"]
        .agg(matches="count", correct="sum")
        .assign(accuracy=lambda x: x["correct"] / x["matches"])
        .reset_index()
    )
    by_league = (
        completed.groupby("league")["correct"]
        .agg(matches="count", correct="sum")
        .assign(accuracy=lambda x: x["correct"] / x["matches"])
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )

    return {
        "total_predictions": len(evaluated),
        "evaluated": len(completed),
        "correct": int(completed["correct"].sum()),
        "accuracy": float(completed["correct"].mean()),
        "by_model": by_model,
        "by_league": by_league,
    }
