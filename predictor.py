"""Wykrywanie kolejki i generowanie prognoz."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from baselines import ensemble_predictions, predict_matches_odds
from compare_models import compare_models, format_comparison_report
from config import (
    COMPARISON_PATH,
    DEFAULT_MODEL,
    LEAGUES,
    OUTPUT_DIR,
    PREDICTIONS_PATH,
)
from database import load_played_matches
from dixon_coles import load_dixon_coles, predict_matches_dc, train_dixon_coles
from features import build_prediction_features, build_training_features
from model import load_model, predict_matches, train_model
from schedule import get_next_round_fixtures

VALID_MODELS = ("auto", "dixon_coles", "xgboost", "odds", "ensemble")


def resolve_model_name(
    model_name: str | None = None,
    *,
    for_predictions: bool = False,
) -> str:
    """Wybiera model: xgboost, dixon_coles, odds, ensemble lub auto."""
    name = (model_name or DEFAULT_MODEL).lower()
    if name != "auto":
        return name

    # Kursy są benchmarkiem historycznym — do prognoz nadchodzących meczów
    # prawie zawsze brakuje kursów w terminarzu API (tylko co.uk ma kilka).
    auto_candidates = (
        ("ensemble", "xgboost", "dixon_coles")
        if for_predictions
        else ("ensemble", "odds", "xgboost", "dixon_coles")
    )

    if COMPARISON_PATH.exists():
        text = COMPARISON_PATH.read_text(encoding="utf-8")
        for candidate in auto_candidates:
            if f"Lepszy model: {candidate}" in text:
                return candidate

    return "ensemble"


def _format_predictions(predictions: pd.DataFrame, model_used: str) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["model"] = model_used
    if "league" not in predictions.columns:
        predictions["league"] = predictions["div"].map(LEAGUES)
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.strftime("%Y-%m-%d")
    output_cols = [
        "model",
        "league",
        "div",
        "round_id",
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
    present = [c for c in output_cols if c in predictions.columns]
    return predictions[present]


def _print_round_summary(fixtures: pd.DataFrame) -> None:
    if fixtures.empty or "round_id" not in fixtures.columns:
        return
    for div in fixtures["div"].unique():
        league = LEAGUES.get(div, div)
        chunk = fixtures[fixtures["div"] == div]
        round_id = int(chunk["round_id"].iloc[0])
        print(f"  {league}: kolejka {round_id} ({len(chunk)} meczow)")


def _merge_fixture_odds(fixtures: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Uzupełnia kursy w fixtures z cech predykcji (historia + terminarz)."""
    enriched = build_prediction_features(fixtures, history)
    out = fixtures.copy()
    for col in ("avg_h", "avg_d", "avg_a"):
        if col in enriched.columns:
            if col not in out.columns:
                out[col] = enriched[col]
            else:
                out[col] = out[col].combine_first(enriched[col])
    return out


def _attach_round_meta(predictions: pd.DataFrame, next_round: pd.DataFrame) -> pd.DataFrame:
    if "round_id" not in next_round.columns:
        return predictions
    keys = ["div", "date", "home_team", "away_team"]
    meta = next_round[keys + ["round_id", "league"]].copy()
    meta["date"] = pd.to_datetime(meta["date"])
    out = predictions.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.merge(meta, on=keys, how="left")


def _predict_with_model(
    selected: str,
    next_round: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    if selected == "dixon_coles":
        try:
            dc_model = load_dixon_coles()
        except FileNotFoundError:
            dc_model, _ = train_dixon_coles(history)
        return predict_matches_dc(dc_model, next_round)

    if selected == "xgboost":
        training = build_training_features(history)
        if training.empty:
            raise RuntimeError("Nie udało się zbudować cech treningowych.")
        try:
            xgb_model = load_model()
        except FileNotFoundError:
            xgb_model, _ = train_model(training)
        features = build_prediction_features(next_round, history)
        return predict_matches(xgb_model, features)

    if selected == "odds":
        with_odds = _merge_fixture_odds(next_round, history)
        preds = predict_matches_odds(with_odds)
        if preds.empty or len(preds) < len(next_round):
            covered = len(preds)
            total = len(next_round)
            print(
                f"  Kursy dostepne tylko dla {covered}/{total} meczow — "
                "przelaczam na ensemble."
            )
            return _predict_with_model("ensemble", next_round, history)
        return preds

    if selected == "ensemble":
        try:
            dc_model = load_dixon_coles()
        except FileNotFoundError:
            dc_model, _ = train_dixon_coles(history)
        training = build_training_features(history)
        if training.empty:
            raise RuntimeError("Nie udało się zbudować cech treningowych.")
        try:
            xgb_model = load_model()
        except FileNotFoundError:
            xgb_model, _ = train_model(training)
        dc_preds = predict_matches_dc(dc_model, next_round)
        features = build_prediction_features(next_round, history)
        xgb_preds = predict_matches(xgb_model, features)
        return ensemble_predictions(dc_preds, xgb_preds)

    raise ValueError(f"Nieznany model: {selected}")


def run_predictions(
    save_path: Path | None = None,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Generuje prognozy najbliższej kolejki w każdej lidze."""
    save_path = save_path or PREDICTIONS_PATH
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history = load_played_matches()
    if history.empty:
        raise RuntimeError(
            "Baza danych jest pusta. Uruchom najpierw: python main.py init"
        )

    selected = resolve_model_name(model_name, for_predictions=True)
    print(f"Model prognoz: {selected}")
    print("Wykrywanie najblizszych kolejek (6 lig)...")
    next_round = get_next_round_fixtures(played=history)
    if next_round.empty:
        print("Brak nadchodzacych meczow w terminarzu.")
        return pd.DataFrame()

    _print_round_summary(next_round)
    print(f"Razem do prognozy: {len(next_round)} meczow")

    predictions = _predict_with_model(selected, next_round, history)
    if predictions.empty:
        return pd.DataFrame()

    predictions = _attach_round_meta(predictions, next_round)
    output = _format_predictions(predictions, selected)
    from prediction_history import archive_predictions

    archive_predictions(output)
    output.to_csv(save_path, index=False, encoding="utf-8-sig")
    return output


def run_model_comparison(save_path: Path | None = None) -> dict:
    """Porównuje wszystkie modele; zapisuje raport."""
    save_path = save_path or COMPARISON_PATH
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history = load_played_matches()
    result = compare_models(history)
    report = format_comparison_report(result)
    save_path.write_text(report, encoding="utf-8")
    return result


def retrain_and_predict(model_name: str | None = None) -> tuple[dict, pd.DataFrame]:
    """Trenuje oba modele od zera i generuje prognozy."""
    history = load_played_matches()
    training = build_training_features(history)
    _, xgb_metrics = train_model(training)
    _, dc_metrics = train_dixon_coles(history)
    comparison = run_model_comparison()
    predictions = run_predictions(model_name=model_name)
    metrics = {
        "xgboost": xgb_metrics,
        "dixon_coles": dc_metrics,
        "comparison": comparison,
    }
    return metrics, predictions
