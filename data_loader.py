"""Pobieranie danych historycznych z football-data.co.uk."""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import requests

from config import (
    BASE_URL,
    FIXTURES_URL,
    LEAGUES,
    MATCH_DB_COLUMNS,
    NUM_SEASONS,
    ODDS_CSV_MAP,
    RAW_DIR,
    STAT_CSV_MAP,
)


def _season_codes(num_seasons: int = NUM_SEASONS) -> list[str]:
    """Generuje kody sezonów, np. 2526, 2425, ..."""
    now = datetime.now()
    start_year = now.year if now.month >= 7 else now.year - 1
    codes = []
    for i in range(num_seasons):
        y1 = start_year - i
        y2 = y1 + 1
        codes.append(f"{y1 % 100:02d}{y2 % 100:02d}")
    return codes


def _download_csv(url: str) -> pd.DataFrame | None:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.content.decode("utf-8-sig", errors="replace")
        if "HomeTeam" not in content and "home_team" not in content.lower():
            return None
        return pd.read_csv(
            io.StringIO(content),
            on_bad_lines="warn",
            engine="python",
        )
    except (
        requests.RequestException,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        ValueError,
    ):
        return None


def _first_available(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Bierze pierwszą dostępną kolumnę z listy kandydatów."""
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index)


def _extract_extra_columns(df: pd.DataFrame) -> pd.DataFrame:
    extra = pd.DataFrame(index=df.index)
    for target, sources in ODDS_CSV_MAP + STAT_CSV_MAP:
        extra[target] = _first_available(df, sources)
    return extra


def _normalize_results(df: pd.DataFrame, div: str, season: str) -> pd.DataFrame:
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    out = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    out.columns = ["date", "home_team", "away_team", "fthg", "ftag", "ftr"]
    out["date"] = pd.to_datetime(out["date"], format="%d/%m/%Y", errors="coerce")
    out["div"] = div
    out["season"] = season

    for col in ["fthg", "ftag"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    extras = _extract_extra_columns(df)
    out = pd.concat([out, extras], axis=1)

    out = out.dropna(subset=["date", "home_team", "away_team"])
    out["ftr"] = out["ftr"].astype(str).str.strip().str.upper()
    out = out[out["ftr"].isin(["H", "D", "A", "NAN"]) | out["ftr"].isna()]
    out.loc[~out["ftr"].isin(["H", "D", "A"]), "ftr"] = None

    return out[MATCH_DB_COLUMNS]


def _normalize_fixtures(raw: pd.DataFrame, leagues: dict[str, str]) -> pd.DataFrame:
    required = {"Div", "Date", "HomeTeam", "AwayTeam"}
    if not required.issubset(raw.columns):
        return pd.DataFrame()

    filtered = raw[raw["Div"].isin(leagues.keys())].copy()
    filtered["date"] = pd.to_datetime(filtered["Date"], format="%d/%m/%Y", errors="coerce")
    filtered = filtered.dropna(subset=["date", "HomeTeam", "AwayTeam"])
    if filtered.empty:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "div": filtered["Div"].values,
            "date": filtered["date"].values,
            "home_team": filtered["HomeTeam"].values,
            "away_team": filtered["AwayTeam"].values,
            "season": _current_season_code(),
            "fthg": None,
            "ftag": None,
            "ftr": None,
        }
    )
    extras = _extract_extra_columns(filtered)
    df = pd.concat([df.reset_index(drop=True), extras.reset_index(drop=True)], axis=1)
    return df[MATCH_DB_COLUMNS].sort_values(["div", "date"])


def download_historical_data(
    leagues: dict[str, str] | None = None,
    num_seasons: int = NUM_SEASONS,
    save_raw: bool = True,
    source: str | None = None,
) -> pd.DataFrame:
    """Pobiera wyniki z ostatnich N sezonów (CSV, API lub hybrid)."""
    from config import DATA_SOURCE

    source = (source or DATA_SOURCE).lower()

    if source == "api":
        from api_loader import download_historical_data_api

        print("Zrodlo danych: football-data.org API")
        return download_historical_data_api(leagues, num_seasons)

    if source == "hybrid":
        print("Zrodlo danych: hybrid (historia CSV + mozliwosc API)")
        return _download_historical_csv(leagues, num_seasons, save_raw)

    print("Zrodlo danych: football-data.co.uk CSV")
    return _download_historical_csv(leagues, num_seasons, save_raw)


def _download_historical_csv(
    leagues: dict[str, str] | None = None,
    num_seasons: int = NUM_SEASONS,
    save_raw: bool = True,
) -> pd.DataFrame:
    """Pobiera wyniki z football-data.co.uk (CSV)."""
    leagues = leagues or LEAGUES
    seasons = _season_codes(num_seasons)
    frames: list[pd.DataFrame] = []

    if save_raw:
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        for div in leagues:
            url = f"{BASE_URL}/{season}/{div}.csv"
            raw = _download_csv(url)
            if raw is None or raw.empty:
                print(f"  Pominięto: {div} {season} (brak danych)")
                continue

            if save_raw:
                raw.to_csv(RAW_DIR / f"{div}_{season}.csv", index=False)

            normalized = _normalize_results(raw, div, season)
            if not normalized.empty:
                frames.append(normalized)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["div", "season", "date", "home_team", "away_team"]
    )
    combined = combined.sort_values(["date", "div"]).reset_index(drop=True)
    return combined


def download_fixtures(leagues: dict[str, str] | None = None, source: str | None = None) -> pd.DataFrame:
    """Pobiera nadchodzące mecze."""
    from config import DATA_SOURCE

    source = (source or DATA_SOURCE).lower()

    if source in ("api", "hybrid"):
        try:
            from api_loader import download_fixtures_api

            fixtures = download_fixtures_api(leagues)
            if not fixtures.empty:
                return fixtures
        except Exception as exc:
            print(f"  API fixtures niedostepne ({exc}) - fallback CSV")

    raw = _download_csv(FIXTURES_URL)
    if raw is None or raw.empty:
        return pd.DataFrame()
    return _normalize_fixtures(raw, leagues or LEAGUES)


def _current_season_code() -> str:
    now = datetime.now()
    start_year = now.year if now.month >= 7 else now.year - 1
    end_year = start_year + 1
    return f"{start_year % 100:02d}{end_year % 100:02d}"


def refresh_current_season(
    leagues: dict[str, str] | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Pobiera bieżący sezon dla aktualizacji bazy."""
    from config import DATA_SOURCE

    source = (source or DATA_SOURCE).lower()

    if source == "api":
        from api_loader import refresh_current_season_api

        return refresh_current_season_api(leagues)

    if source == "hybrid":
        try:
            from api_loader import refresh_current_season_api

            api_df = refresh_current_season_api(leagues)
            csv_df = _refresh_current_season_csv(leagues)
            if api_df.empty:
                return csv_df
            if csv_df.empty:
                return api_df
            combined = pd.concat([csv_df, api_df], ignore_index=True)
            return combined.drop_duplicates(
                subset=["div", "season", "date", "home_team", "away_team"],
                keep="last",
            )
        except Exception as exc:
            print(f"  API refresh niedostepne ({exc}) - fallback CSV")
            return _refresh_current_season_csv(leagues)

    return _refresh_current_season_csv(leagues)


def fetch_finished_results(
    leagues: dict[str, str] | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Rozegrane mecze bieżącego sezonu (CSV + API w trybie hybrid)."""
    current = refresh_current_season(leagues=leagues, source=source or "hybrid")
    if current.empty:
        return current
    return current[current["ftr"].isin(["H", "D", "A"])].copy()


def _refresh_current_season_csv(
    leagues: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Bieżący sezon z football-data.co.uk."""
    leagues = leagues or LEAGUES
    season = _current_season_code()
    frames: list[pd.DataFrame] = []

    for div in leagues:
        url = f"{BASE_URL}/{season}/{div}.csv"
        raw = _download_csv(url)
        if raw is None or raw.empty:
            continue
        normalized = _normalize_results(raw, div, season)
        if not normalized.empty:
            frames.append(normalized)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
