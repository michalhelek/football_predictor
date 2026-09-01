"""Aktualizacja bazy po rozegraniu kolejki i ponowny trening modelu."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import METADATA_PATH, PREDICTIONS_PATH
from data_loader import download_historical_data, fetch_finished_results
from database import load_all_matches, load_played_matches, upsert_matches
from predictor import retrain_and_predict


def _load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _save_metadata(data: dict) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def initialize_database() -> int:
    """Pobiera 5 lat historii i zapisuje do bazy."""
    historical = download_historical_data()
    if historical.empty:
        raise RuntimeError("Nie udało się pobrać danych historycznych.")

    added = upsert_matches(historical)
    _save_metadata(
        {
            "initialized_at": datetime.now().isoformat(),
            "total_matches": len(historical),
            "last_update": datetime.now().isoformat(),
        }
    )
    return added


def _predicted_round_matches() -> pd.DataFrame | None:
    """Wczytuje ostatnie prognozy, aby sprawdzić status kolejki."""
    if not PREDICTIONS_PATH.exists():
        return None

    preds = pd.read_csv(PREDICTIONS_PATH, encoding="utf-8-sig")
    if preds.empty:
        return None

    preds["date"] = pd.to_datetime(preds["date"])
    return preds


def _prediction_keys(predictions: pd.DataFrame) -> list[tuple]:
    preds = predictions.copy()
    preds["date"] = pd.to_datetime(preds["date"])
    return [
        (row.div, row.date.strftime("%Y-%m-%d"), row.home_team, row.away_team)
        for row in preds.itertuples()
    ]


def is_predicted_round_complete(
    predictions: pd.DataFrame | None = None,
    played: pd.DataFrame | None = None,
) -> bool:
    """Sprawdza, czy wszystkie prognozowane mecze mają już wyniki w bazie."""
    predictions = predictions if predictions is not None else _predicted_round_matches()
    if predictions is None or predictions.empty:
        return False

    played = played if played is not None else load_played_matches()
    if played.empty:
        return False

    played_keys = set(
        zip(
            played["div"],
            played["date"].dt.strftime("%Y-%m-%d"),
            played["home_team"],
            played["away_team"],
        )
    )
    pred_keys = _prediction_keys(predictions)
    return all(key in played_keys for key in pred_keys)

def update_after_round(force: bool = False) -> dict:
    """
    Pobiera najnowsze wyniki, dopisuje do bazy, trenuje model i generuje prognozy.

    force=True pomija sprawdzenie, czy kolejka została rozegrana.
    """
    predictions = _predicted_round_matches()
    if not force and not is_predicted_round_complete(predictions):
        pending = 0
        if predictions is not None:
            played = load_played_matches()
            played_keys = set(
                zip(
                    played["div"],
                    played["date"].dt.strftime("%Y-%m-%d"),
                    played["home_team"],
                    played["away_team"],
                )
            )
            pred_keys = _prediction_keys(predictions)
            pending = sum(1 for k in pred_keys if k not in played_keys)

        return {
            "updated": False,
            "reason": f"Kolejka nie została jeszcze w pełni rozegrana ({pending} meczów bez wyniku).",
        }

    # Wyniki bieżącego sezonu: hybrid (co.uk + API) — samo CSV ma opóźnienie i luki
    print("Synchronizacja wynikow biezacego sezonu (CSV + API)...")
    finished = fetch_finished_results(source="hybrid")
    added_current = upsert_matches(finished) if not finished.empty else 0
    if not finished.empty:
        latest = pd.to_datetime(finished["date"]).max().date()
        print(f"  Rozegrane w paczce: {len(finished)} | ostatni mecz: {latest}")

    # Historia 5 sezonów z co.uk (kursy + statystyki) — aktualizacja istniejących wierszy
    print("Odswiezanie historii z football-data.co.uk...")
    historical = download_historical_data(save_raw=False)
    added_hist = upsert_matches(historical) if not historical.empty else 0

    metrics, new_predictions = retrain_and_predict()

    meta = _load_metadata()
    meta["last_update"] = datetime.now().isoformat()
    meta["last_retrain"] = datetime.now().isoformat()
    meta["matches_in_db"] = len(load_played_matches())
    meta["played_in_db"] = meta["matches_in_db"]
    meta["last_metrics"] = {
        "comparison": metrics.get("comparison", {}),
        "xgboost": metrics.get("xgboost", {}),
        "dixon_coles": metrics.get("dixon_coles", {}),
    }
    _save_metadata(meta)

    return {
        "updated": True,
        "added_current_season": added_current,
        "added_historical": added_hist,
        "finished_synced": len(finished) if not finished.empty else 0,
        "metrics": metrics,
        "new_predictions_count": len(new_predictions),
        "predictions_path": str(PREDICTIONS_PATH),
    }
