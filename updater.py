"""Aktualizacja bazy po rozegraniu kolejki i ponowny trening modelu."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import METADATA_PATH, PREDICTIONS_PATH, get_data_source
from data_loader import download_historical_data, fetch_finished_results
from database import load_played_matches, restore_from_seed, seed_database_available, upsert_matches
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


def initialize_database(save_raw: bool = True) -> dict:
    """
    Ładuje historię meczów do bazy.

    Streamlit Cloud: preferuje matches_seed.db (8418 meczów + kursy z co.uk),
    potem dogrywa najnowsze wyniki z API. Samo API nie ma kursów bukmacherskich.
    """
    from api_loader import FootballDataOrgError, validate_api_token
    from config import NUM_SEASONS, get_data_source

    source = get_data_source()

    if seed_database_available():
        count = restore_from_seed()
        updates = fetch_finished_results(source="hybrid")
        added = upsert_matches(updates) if not updates.empty else 0
        _save_metadata(
            {
                "initialized_at": datetime.now().isoformat(),
                "total_matches": count,
                "source": "matches_seed.db + API",
                "last_update": datetime.now().isoformat(),
            }
        )
        played = load_played_matches()
        return {
            "added": count,
            "with_odds": int(played["avg_h"].notna().sum()),
            "source": "seed",
            "api_updates": added,
        }

    if source in ("api", "hybrid"):
        ok, msg = validate_api_token()
        if not ok:
            raise RuntimeError(msg)

    try:
        historical = download_historical_data(
            save_raw=save_raw,
            source=source,
            num_seasons=NUM_SEASONS,
        )
    except FootballDataOrgError as exc:
        raise RuntimeError(str(exc)) from exc

    if historical.empty and source != "api":
        try:
            historical = download_historical_data(
                save_raw=save_raw,
                source="api",
                num_seasons=NUM_SEASONS,
            )
        except FootballDataOrgError as exc:
            raise RuntimeError(str(exc)) from exc

    if historical.empty:
        raise RuntimeError(
            "Nie udało się pobrać meczów. Dodaj plik data/matches_seed.db do repo "
            "(pełna historia z kursami) lub poczekaj aż football-data.co.uk wróci online."
        )

    upsert_matches(historical)
    played = load_played_matches()
    _save_metadata(
        {
            "initialized_at": datetime.now().isoformat(),
            "total_matches": len(historical),
            "source": source,
            "last_update": datetime.now().isoformat(),
        }
    )
    return {
        "added": len(historical),
        "with_odds": int(played["avg_h"].notna().sum()),
        "source": source,
        "api_updates": 0,
    }


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


def _sync_historical_for_update(played_count: int) -> tuple[pd.DataFrame, str]:
    """
    Przy aktualizacji kolejki nie pobieramy 5 sezonów z API (30 zapytań → 429).
    Historia + kursy są już w bazie seed; wystarczy bieżący sezon + opcjonalnie CSV.
    """
    source = get_data_source()

    if source in ("csv", "hybrid"):
        print("Odswiezanie kursow i statystyk z football-data.co.uk...")
        historical = download_historical_data(save_raw=False, source="csv")
        if not historical.empty:
            return historical, "csv"
        if source == "csv":
            return historical, "csv"
        print("  CSV niedostepne — pominięto odswiezanie historii.")

    if played_count >= 1000:
        print(
            "  Pominięto pełną historię API — baza zawiera już dane historyczne "
            f"({played_count} meczów)."
        )
        return pd.DataFrame(), "skipped"

    print("  Baza bez pełnej historii — uzupełnienie bieżącego sezonu z API...")
    from api_loader import refresh_current_season_api

    current = refresh_current_season_api()
    if current.empty:
        return current, "api_current"
    finished = current[current["ftr"].isin(["H", "D", "A"])].copy()
    return finished, "api_current"


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

    played_before = load_played_matches()
    played_count = len(played_before)

    # Wyniki bieżącego sezonu: hybrid (co.uk + API) — ok. 6 zapytań API, nie 30
    print("Synchronizacja wynikow biezacego sezonu (CSV + API)...")
    finished = fetch_finished_results(source="hybrid")
    added_current = upsert_matches(finished) if not finished.empty else 0
    if not finished.empty:
        latest = pd.to_datetime(finished["date"]).max().date()
        print(f"  Rozegrane w paczce: {len(finished)} | ostatni mecz: {latest}")

    historical, hist_source = _sync_historical_for_update(played_count)
    added_hist = upsert_matches(historical) if not historical.empty else 0
    if added_hist:
        print(f"  Zaktualizowano wiersze historii ({hist_source}): {added_hist}")

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
