"""
Pobieranie danych z API football-data.org (v4).

Wymaga tokenu: https://www.football-data.org/client/register
Ustaw zmienną środowiskową FOOTBALL_DATA_ORG_TOKEN lub plik data/.api_token
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
import requests

from config import (
    API_LEAGUE_CODES,
    API_REQUEST_DELAY_SEC,
    FOOTBALL_DATA_ORG_API,
    MATCH_DB_COLUMNS,
    NUM_SEASONS,
)


class FootballDataOrgError(Exception):
    pass


def get_api_token() -> str:
    """Token z env lub pliku data/.api_token (jedna linia)."""
    token = os.getenv("FOOTBALL_DATA_ORG_TOKEN", "").strip()
    if token:
        return token

    from config import API_TOKEN_FILE

    if API_TOKEN_FILE.exists():
        token = API_TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            return token

    raise FootballDataOrgError(
        "Brak tokenu API. Ustaw FOOTBALL_DATA_ORG_TOKEN lub zapisz token w data/.api_token"
    )


def _api_headers() -> dict[str, str]:
    return {"X-Auth-Token": get_api_token()}


def _season_start_years(num_seasons: int = NUM_SEASONS) -> list[int]:
    now = datetime.now()
    start_year = now.year if now.month >= 7 else now.year - 1
    return [start_year - i for i in range(num_seasons)]


def _season_code(year: int) -> str:
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def _winner_to_ftr(winner: str | None) -> str | None:
    mapping = {
        "HOME_TEAM": "H",
        "AWAY_TEAM": "A",
        "DRAW": "D",
    }
    return mapping.get(winner)


def _normalize_match(match: dict, div: str, season_year: int) -> dict | None:
    home = match.get("homeTeam") or {}
    away = match.get("awayTeam") or {}
    score = match.get("score") or {}
    full_time = score.get("fullTime") or {}

    home_name = home.get("name") or home.get("shortName")
    away_name = away.get("name") or away.get("shortName")
    if not home_name or not away_name:
        return None

    utc_date = match.get("utcDate")
    if not utc_date:
        return None
    date = pd.to_datetime(utc_date, utc=True).tz_convert(None)

    status = match.get("status", "")
    fthg = full_time.get("home")
    ftag = full_time.get("away")
    winner = score.get("winner")
    ftr = _winner_to_ftr(winner)

    if status != "FINISHED":
        fthg, ftag, ftr = None, None, None
    elif fthg is None or ftag is None:
        ftr = None

    row = {
        "div": div,
        "season": _season_code(season_year),
        "date": date,
        "home_team": home_name,
        "away_team": away_name,
        "fthg": fthg,
        "ftag": ftag,
        "ftr": ftr,
        "matchday": match.get("matchday"),
        "avg_h": None,
        "avg_d": None,
        "avg_a": None,
        "home_shots": None,
        "away_shots": None,
        "home_sot": None,
        "away_sot": None,
        "home_corners": None,
        "away_corners": None,
        "home_fouls": None,
        "away_fouls": None,
    }
    return row


def _fetch_competition_matches(
    api_code: str,
    season_year: int,
    status: str | None = None,
) -> list[dict]:
    url = f"{FOOTBALL_DATA_ORG_API}/competitions/{api_code}/matches"
    params: dict[str, str | int] = {"season": season_year}
    if status:
        params["status"] = status

    response = requests.get(url, headers=_api_headers(), params=params, timeout=30)
    if response.status_code == 403:
        raise FootballDataOrgError(
            "403 Forbidden - sprawdź token lub dostęp do tej ligi w planie API."
        )
    if response.status_code == 429:
        raise FootballDataOrgError(
            "429 Too Many Requests - przekroczono limit zapytań API (poczekaj chwilę)."
        )
    response.raise_for_status()

    payload = response.json()
    return payload.get("matches", [])


def download_historical_data_api(
    leagues: dict[str, str] | None = None,
    num_seasons: int = NUM_SEASONS,
) -> pd.DataFrame:
    """Pobiera zakończone mecze z ostatnich N sezonów przez API."""
    from config import LEAGUES

    leagues = leagues or LEAGUES
    seasons = _season_start_years(num_seasons)
    rows: list[dict] = []

    for div in leagues:
        api_code = API_LEAGUE_CODES.get(div)
        if not api_code:
            print(f"  Pominięto {div}: brak mapowania API")
            continue

        for season_year in seasons:
            print(f"  API: {div} sezon {season_year}/{season_year + 1}...", end=" ")
            try:
                matches = _fetch_competition_matches(
                    api_code, season_year, status="FINISHED"
                )
            except requests.RequestException as exc:
                print(f"blad ({exc})")
                time.sleep(API_REQUEST_DELAY_SEC)
                continue

            count = 0
            for match in matches:
                row = _normalize_match(match, div, season_year)
                if row and row["ftr"] in ("H", "D", "A"):
                    rows.append(row)
                    count += 1
            print(f"{count} meczow")
            time.sleep(API_REQUEST_DELAY_SEC)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(
        subset=["div", "season", "date", "home_team", "away_team"]
    )
    return df[MATCH_DB_COLUMNS].sort_values(["date", "div"]).reset_index(drop=True)


def download_fixtures_api(leagues: dict[str, str] | None = None) -> pd.DataFrame:
    """Pobiera nadchodzące mecze (SCHEDULED/TIMED) z API."""
    from config import LEAGUES

    leagues = leagues or LEAGUES
    season_year = _season_start_years(1)[0]
    rows: list[dict] = []

    for div in leagues:
        api_code = API_LEAGUE_CODES.get(div)
        if not api_code:
            continue

        for status in ("SCHEDULED", "TIMED"):
            try:
                matches = _fetch_competition_matches(api_code, season_year, status=status)
            except requests.RequestException:
                time.sleep(API_REQUEST_DELAY_SEC)
                continue

            for match in matches:
                row = _normalize_match(match, div, season_year)
                if row and row["ftr"] is None:
                    rows.append(row)
            time.sleep(API_REQUEST_DELAY_SEC)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(
        subset=["div", "date", "home_team", "away_team"]
    )
    return df[MATCH_DB_COLUMNS].sort_values(["div", "date"]).reset_index(drop=True)


def refresh_current_season_api(
    leagues: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Bieżący sezon - wszystkie mecze z wynikiem."""
    from config import LEAGUES

    leagues = leagues or LEAGUES
    season_year = _season_start_years(1)[0]
    rows: list[dict] = []

    for div in leagues:
        api_code = API_LEAGUE_CODES.get(div)
        if not api_code:
            continue
        try:
            matches = _fetch_competition_matches(api_code, season_year)
        except requests.RequestException:
            time.sleep(API_REQUEST_DELAY_SEC)
            continue

        for match in matches:
            row = _normalize_match(match, div, season_year)
            if row:
                rows.append(row)
        time.sleep(API_REQUEST_DELAY_SEC)

    if not rows:
        return pd.DataFrame()

    return _schedule_columns(pd.DataFrame(rows))


def _schedule_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Zwraca kolumny terminarza, zachowując matchday z API jeśli jest."""
    cols = [*MATCH_DB_COLUMNS]
    if "matchday" in df.columns:
        cols.append("matchday")
    present = [c for c in cols if c in df.columns]
    return df[present]


def download_season_schedule_api(
    leagues: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Pełny terminarz bieżącego sezonu z football-data.org (wszystkie kolejki)."""
    return refresh_current_season_api(leagues)
