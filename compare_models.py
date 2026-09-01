"""Porównanie modeli prognozujących na danych z football-data.co.uk."""

from __future__ import annotations

import pandas as pd

from baselines import (
    ensemble_predictions,
    evaluate_predictions,
    predict_matches_odds,
)
from dixon_coles import predict_matches_dc, train_dixon_coles
from features import build_prediction_features, build_training_features
from mlp_model import predict_matches_mlp, train_mlp_model
from model import predict_matches, train_model


def _chronological_split(
    matches: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    played = matches[matches["ftr"].isin(["H", "D", "A"])].sort_values("date")
    split_idx = int(len(played) * (1 - test_fraction))
    return played.iloc[:split_idx].copy(), played.iloc[split_idx:].copy()


def _evaluate_xgboost(model, test_matches: pd.DataFrame, train_matches: pd.DataFrame) -> dict:
    features = build_prediction_features(test_matches, train_matches)
    if features.empty:
        return {"accuracy": 0.0, "log_loss": None, "samples": 0, "report": ""}
    preds = predict_matches(model, features)
    return evaluate_predictions(preds, test_matches)


def _evaluate_mlp(model, test_matches: pd.DataFrame, train_matches: pd.DataFrame) -> dict:
    features = build_prediction_features(test_matches, train_matches)
    if features.empty:
        return {"accuracy": 0.0, "log_loss": None, "samples": 0, "report": ""}
    preds = predict_matches_mlp(model, features)
    return evaluate_predictions(preds, test_matches)


def _evaluate_dixon_coles(model, test_matches: pd.DataFrame) -> dict:
    preds = predict_matches_dc(model, test_matches)
    return evaluate_predictions(preds, test_matches)


def _evaluate_odds(test_matches: pd.DataFrame) -> dict:
    preds = predict_matches_odds(test_matches)
    return evaluate_predictions(preds, test_matches)


def _evaluate_ensemble(
    dc_model,
    xgb_model,
    test_matches: pd.DataFrame,
    train_matches: pd.DataFrame,
) -> dict:
    dc_preds = predict_matches_dc(dc_model, test_matches)
    features = build_prediction_features(test_matches, train_matches)
    if features.empty:
        return {"accuracy": 0.0, "log_loss": None, "samples": 0, "report": ""}
    xgb_preds = predict_matches(xgb_model, features)
    combined = ensemble_predictions(dc_preds, xgb_preds)
    return evaluate_predictions(combined, test_matches)


def _pick_winner(metrics: dict[str, dict]) -> str:
    best_name = "remis"
    best_acc = -1.0
    for name, data in metrics.items():
        acc = data.get("accuracy", 0.0)
        if acc > best_acc:
            best_acc = acc
            best_name = name
    return best_name


def compare_models(
    matches: pd.DataFrame,
    test_fraction: float = 0.2,
) -> dict:
    """Porównuje DC, XGBoost, MLP, kursy bukmacherskie i ensemble."""
    train, test = _chronological_split(matches, test_fraction)

    dc_model, dc_train_info = train_dixon_coles(train)
    training = build_training_features(train)
    xgb_model, xgb_train_metrics = train_model(training)
    mlp_model, mlp_train_metrics = train_mlp_model(training)

    dc_metrics = _evaluate_dixon_coles(dc_model, test)
    xgb_metrics = _evaluate_xgboost(xgb_model, test, train)
    mlp_metrics = _evaluate_mlp(mlp_model, test, train)
    odds_metrics = _evaluate_odds(test)
    ensemble_metrics = _evaluate_ensemble(dc_model, xgb_model, test, train)

    scored = {
        "dixon_coles": dc_metrics,
        "xgboost": xgb_metrics,
        "mlp": mlp_metrics,
        "odds": odds_metrics,
        "ensemble": ensemble_metrics,
    }
    winner = _pick_winner(scored)

    return {
        "train_matches": len(train),
        "test_matches": len(test),
        "dixon_coles": {**dc_metrics, "train_info": dc_train_info},
        "xgboost": {**xgb_metrics, "train_seconds": xgb_train_metrics.get("train_seconds")},
        "mlp": {**mlp_metrics, "train_seconds": mlp_train_metrics.get("train_seconds")},
        "odds": odds_metrics,
        "ensemble": ensemble_metrics,
        "winner": winner,
        "summary": (
            f"DC: {dc_metrics['accuracy']:.1%} | XGB: {xgb_metrics['accuracy']:.1%} | "
            f"MLP: {mlp_metrics['accuracy']:.1%} | Kursy: {odds_metrics['accuracy']:.1%} | "
            f"Ensemble: {ensemble_metrics['accuracy']:.1%} | Lepszy: {winner}"
        ),
    }


def format_comparison_report(result: dict) -> str:
    dc = result["dixon_coles"]
    xgb = result["xgboost"]
    mlp = result["mlp"]
    odds = result["odds"]
    ens = result["ensemble"]

    def fmt_block(title: str, data: dict) -> list[str]:
        block = [
            f"--- {title} ---",
            f"Dokladnosc:  {data['accuracy']:.1%}",
            f"Log-loss:    {data['log_loss']:.4f}" if data.get("log_loss") else "Log-loss:    n/d",
            f"Mecze ocenione: {data.get('samples', 0)}",
        ]
        if data.get("train_seconds") is not None:
            block.append(f"Czas treningu: {data['train_seconds']}s")
        block.append("")
        if data.get("report"):
            block.extend([data["report"], ""])
        return block

    lines = [
        "=== Porownanie modeli (football-data.co.uk) ===",
        f"Zbior treningowy: {result['train_matches']} meczow",
        f"Zbior testowy:    {result['test_matches']} meczow (chronologicznie)",
        "",
        *fmt_block("Dixon-Coles (bramki + druzyny)", dc),
        *fmt_block("XGBoost (Elo + forma + kursy + statystyki)", xgb),
        *fmt_block("MLP (siec neuronowa, te same cechy co XGBoost)", mlp),
        *fmt_block("Kursy bukmacherskie (benchmark rynku)", odds),
        *fmt_block("Ensemble (srednia DC + XGBoost)", ens),
        f"Lepszy model: {result['winner']}",
    ]
    return "\n".join(lines)
