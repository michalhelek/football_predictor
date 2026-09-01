"""Terminarz bieżącego sezonu — rozegrane i nadchodzące mecze."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from config import LEAGUES, MATCH_DB_COLUMNS, ROUND_MAX_GAP_DAYS
from data_loader import (
    _current_season_code,
    _refresh_current_season_csv,
    download_fixtures,
)

_TEAM_SUFFIXES = (
    " football club",
    " fc",
    " afc",
    " cf",
    " sc",
    " sv",
    " bv",
    " ud",
)

_TEAM_PREFIXES = (
    "deportivo ",
    "real ",
    "rcd ",
    "rc ",
    "club ",
    "atletico ",
    "athletic ",
    "sporting ",
    "rayo ",
    "dep ",
)

# Różnice pisowni API (football-data.org) vs co.uk (skróty fixtures)
_TEAM_ALIASES = {
    "espanyol": "espanol",
    "ath bilbao": "athletic bilbao",
    "ath madrid": "athletic madrid",
    "man united": "manchester united",
    "man city": "manchester city",
    "newcastle": "newcastle united",
    "santander": "racing club",
}


def assign_rounds(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Grupuje mecze w kolejki na podstawie przerw czasowych w ramach ligi."""
    if fixtures.empty:
        return fixtures

    df = fixtures.sort_values(["div", "date"]).copy()
    df["round_id"] = 0

    for div in df["div"].unique():
        mask = df["div"] == div
        league_fixtures = df.loc[mask]
        round_num = 0
        prev_date = None
        round_ids = []

        for date in league_fixtures["date"]:
            if prev_date is None or (date - prev_date).days > ROUND_MAX_GAP_DAYS:
                round_num += 1
            round_ids.append(round_num)
            prev_date = date

        df.loc[mask, "round_id"] = round_ids

    return df


_assign_rounds = assign_rounds  # alias wewnętrzny


def _numeric_matchday(series: pd.Series) -> pd.Series:
    """Matchday z API jako float; brak wartości -> NaN."""
    return pd.to_numeric(series, errors="coerce")


def _round_id_from_matchday(matchday: pd.Series, default: int) -> pd.Series:
    """Bezpieczna konwersja matchday -> round_id (bez IntCastingNaNError)."""
    numeric = _numeric_matchday(matchday)
    return numeric.fillna(default).astype(int)


def _teams_in_league(history: pd.DataFrame, div: str) -> set[str]:
    league = history[history["div"] == div]
    if league.empty:
        return set()
    return set(league["home_team"]).union(set(league["away_team"]))


def resolve_fixture_teams(
    fixtures: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Mapuje nazwy drużyn z API na nazwy używane w bazie (co.uk)."""
    if fixtures.empty or history.empty:
        return fixtures

    df = fixtures.copy()
    cache: dict[tuple[str, str], str] = {}

    def resolve(name: str, div: str) -> str:
        key = (div, name)
        if key in cache:
            return cache[key]
        teams = _teams_in_league(history, div)
        if name in teams:
            cache[key] = name
            return name
        norm = _norm_team(name)
        matches = [team for team in teams if _norm_team(team) == norm]
        if matches:
            # Preferuj pełną nazwę z API/bazy zamiast skrótu co.uk (Dep., Vallecano…)
            canonical = max(matches, key=len)
            cache[key] = canonical
            return canonical
        cache[key] = name
        return name

    df["home_team"] = df.apply(lambda r: resolve(r["home_team"], r["div"]), axis=1)
    df["away_team"] = df.apply(lambda r: resolve(r["away_team"], r["div"]), axis=1)
    return df


def _assign_round_ids(schedule: pd.DataFrame) -> pd.DataFrame:
    """Numer kolejki: oficjalny matchday z API, fallback heurystyka dat."""
    df = schedule.copy()
    if "matchday" in df.columns and df["matchday"].notna().any():
        df["round_id"] = _numeric_matchday(df["matchday"])
        missing = df["round_id"].isna()
        if missing.any():
            fallback = assign_rounds(df.loc[missing].copy())
            df.loc[missing, "round_id"] = fallback["round_id"].values
        if df["round_id"].isna().any():
            df["round_id"] = df["round_id"].fillna(1)
        df["round_id"] = df["round_id"].astype(int)
        return df
    return assign_rounds(df)


def _select_next_round_for_league(league_up: pd.DataFrame) -> pd.DataFrame:
    """
    Wybiera mecze najbliższej kolejki w lidze.

    Gdy liga ma oficjalny matchday (API): bierze wszystkie nadchodzące mecze
    do daty ostatniego meczu najniższego numeru kolejki. Dzięki temu łapie
    też mecze kolejki N+1 rozgrywane przed zakończeniem kolejki N
    (np. Elche — Barcelona w La Liga).
    """
    if league_up.empty:
        return league_up

    if "matchday" in league_up.columns and league_up["matchday"].notna().any():
        matchdays = _numeric_matchday(league_up["matchday"])
        next_matchday = int(matchdays.min())
        deadline = league_up.loc[matchdays == next_matchday, "date"].max()
        round_matches = league_up[league_up["date"] <= deadline].copy()
        round_matches["round_id"] = _round_id_from_matchday(
            round_matches["matchday"],
            default=next_matchday,
        )
        return round_matches

    league_up = assign_rounds(league_up.sort_values("date"))
    next_round_id = int(league_up["round_id"].min())
    return league_up[league_up["round_id"] == next_round_id].copy()


def get_next_round_fixtures(
    fixtures: pd.DataFrame | None = None,
    reference_date=None,
    played: pd.DataFrame | None = None,
    prefer_api: bool = True,
) -> pd.DataFrame:
    """
    Mecze najbliższej kolejki w każdej lidze (osobno per liga).
    Domyślnie korzysta z pełnego terminarza API + co.uk.
    """
    from datetime import datetime

    if fixtures is not None:
        return _get_next_round_from_fixtures(fixtures, reference_date, played)

    reference_date = reference_date or datetime.now()
    schedule, _ = load_season_schedule(played_db=played, prefer_api=prefer_api)
    if schedule.empty:
        return schedule

    upcoming = schedule[schedule["status"] == "Zaplanowany"].copy()
    upcoming["date"] = pd.to_datetime(upcoming["date"])
    today = pd.Timestamp(reference_date).normalize()
    future = upcoming[upcoming["date"] >= today]
    if not future.empty:
        upcoming = future

    if upcoming.empty:
        return pd.DataFrame()

    chunks: list[pd.DataFrame] = []
    for div in LEAGUES:
        league_up = upcoming[upcoming["div"] == div].sort_values("date")
        if league_up.empty:
            continue
        chunks.append(_select_next_round_for_league(league_up))

    if not chunks:
        return pd.DataFrame()

    merged = pd.concat(chunks, ignore_index=True)

    if played is not None and not played.empty:
        merged = resolve_fixture_teams(merged, played)

    for col in MATCH_DB_COLUMNS:
        if col not in merged.columns:
            merged[col] = None

    cols = [*MATCH_DB_COLUMNS, "round_id", "league"]
    if "matchday" in merged.columns:
        cols.append("matchday")
    extra = [c for c in cols if c in merged.columns]
    return merged[extra].sort_values(["div", "date"]).reset_index(drop=True)


def _get_next_round_from_fixtures(
    fixtures: pd.DataFrame,
    reference_date,
    played: pd.DataFrame | None,
) -> pd.DataFrame:
    """Wykrywanie kolejki z przekazanego DataFrame (np. notebook)."""
    from datetime import datetime

    if fixtures.empty:
        return fixtures

    if played is None:
        from database import load_played_matches

        played = load_played_matches()

    played_keys = set()
    if not played.empty:
        played["date"] = pd.to_datetime(played["date"])
        for row in played.itertuples():
            played_keys.add(
                (
                    row.div,
                    row.date.strftime("%Y-%m-%d"),
                    row.home_team,
                    row.away_team,
                )
            )

    def is_unplayed(row) -> bool:
        key = (row.div, row.date.strftime("%Y-%m-%d"), row.home_team, row.away_team)
        return key not in played_keys

    upcoming = fixtures[fixtures.apply(is_unplayed, axis=1)].copy()
    if upcoming.empty:
        return upcoming

    upcoming["date"] = pd.to_datetime(upcoming["date"])
    if "matchday" in upcoming.columns and upcoming["matchday"].notna().any():
        chunks: list[pd.DataFrame] = []
        for div in upcoming["div"].unique():
            league_up = upcoming[upcoming["div"] == div].sort_values("date")
            chunks.append(_select_next_round_for_league(league_up))
        return pd.concat(chunks, ignore_index=True).sort_values(["div", "date"]).reset_index(drop=True)

    upcoming = assign_rounds(upcoming)
    rounds = (
        upcoming.groupby(["div", "round_id"])["date"]
        .min()
        .reset_index()
        .rename(columns={"date": "round_start"})
    )
    next_rounds = (
        rounds.sort_values(["div", "round_start"])
        .groupby("div")
        .first()
        .reset_index()[["div", "round_id"]]
    )
    merged = upcoming.merge(next_rounds, on=["div", "round_id"])
    return merged.sort_values(["div", "date"]).reset_index(drop=True)


def _norm_team(name: str) -> str:
    """Uproszczone porównywanie nazw drużyn (API vs CSV)."""
    text = unicodedata.normalize("NFKD", str(name))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    changed = True
    while changed:
        changed = False
        for prefix in _TEAM_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
        for suffix in _TEAM_SUFFIXES:
            if text.endswith(suffix):
                text = text[: -len(suffix)].strip()
                changed = True
    text = re.sub(r"\sde\s+[a-z]+(?:\s+[a-z]+)?$", "", text).strip()
    # co.uk: „Dep. A Coruna” → „a coruna” → „la coruna” (jak w API)
    text = re.sub(r"^a\s+", "la ", text).strip()
    return _TEAM_ALIASES.get(text, text)


def _match_key(row: pd.Series) -> tuple:
    return (
        row["div"],
        pd.Timestamp(row["date"]).date(),
        _norm_team(row["home_team"]),
        _norm_team(row["away_team"]),
    )


def _row_priority(row: pd.Series) -> int:
    score = 0
    if row.get("ftr") in ("H", "D", "A"):
        score += 8
    if pd.notna(row.get("fthg")) and pd.notna(row.get("ftag")):
        score += 4
    if pd.notna(row.get("avg_h")):
        score += 2
    if pd.notna(row.get("matchday")):
        score += 1
    return score


def _merge_schedule_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Łączy terminarze z wielu źródeł; wybiera najpełniejszy wiersz meczu."""
    valid = [df for df in frames if not df.empty]
    if not valid:
        return pd.DataFrame()

    combined = pd.concat(valid, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined["_key"] = combined.apply(_match_key, axis=1)
    combined["_prio"] = combined.apply(_row_priority, axis=1)
    combined = combined.sort_values(["_key", "_prio"])
    combined = combined.drop_duplicates(subset=["_key"], keep="last")
    return combined.drop(columns=["_key", "_prio"])


def _load_csv_schedule(
    leagues: dict[str, str],
    season_code: str,
) -> pd.DataFrame:
    season_df = _refresh_current_season_csv(leagues)
    if not season_df.empty:
        season_df = season_df[season_df["season"] == season_code]

    fixtures_df = download_fixtures(leagues)
    if not fixtures_df.empty:
        fixtures_df = fixtures_df[fixtures_df["season"] == season_code]

    return _merge_schedule_frames([season_df, fixtures_df])


def _load_api_schedule(leagues: dict[str, str]) -> pd.DataFrame:
    from api_loader import download_season_schedule_api

    return download_season_schedule_api(leagues)


def _apply_db_results(schedule: pd.DataFrame, played_db: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty or played_db.empty:
        return schedule

    db = played_db.copy()
    db["date"] = pd.to_datetime(db["date"])
    db = db[db["ftr"].isin(["H", "D", "A"])]
    cols = [c for c in schedule.columns if c in db.columns]
    return _merge_schedule_frames([schedule, db[cols]])


def _format_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty:
        return schedule

    df = schedule.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = _assign_round_ids(df.sort_values(["div", "date"]))
    df["league"] = df["div"].map(LEAGUES)
    df["status"] = df["ftr"].apply(
        lambda x: "Rozegrany" if x in ("H", "D", "A") else "Zaplanowany"
    )
    df["score"] = df.apply(
        lambda r: (
            f"{int(r['fthg'])}:{int(r['ftag'])}"
            if pd.notna(r["fthg"]) and pd.notna(r["ftag"])
            else "—"
        ),
        axis=1,
    )
    df["result_label"] = df["ftr"].map({"H": "H", "D": "D", "A": "A"}).fillna("—")
    return df.sort_values(["div", "date", "round_id"]).reset_index(drop=True)


def load_season_schedule(
    leagues: dict[str, str] | None = None,
    played_db: pd.DataFrame | None = None,
    prefer_api: bool = True,
) -> tuple[pd.DataFrame, str]:
    """
    Terminarz bieżącego sezonu.

    Returns:
        (schedule_df, source_label)
    """
    leagues = leagues or LEAGUES
    season_code = _current_season_code()
    csv_schedule = _load_csv_schedule(leagues, season_code)
    source = "football-data.co.uk (ograniczony)"

    schedule = csv_schedule
    if prefer_api:
        try:
            api_schedule = _load_api_schedule(leagues)
            if not api_schedule.empty:
                schedule = _merge_schedule_frames([api_schedule, csv_schedule])
                source = "football-data.org + co.uk"
        except Exception:
            pass

    if played_db is not None:
        db_season = played_db.copy()
        if not db_season.empty and "season" in db_season.columns:
            db_season = db_season[db_season["season"] == season_code]
        schedule = _apply_db_results(schedule, db_season)
        if not played_db.empty:
            schedule = resolve_fixture_teams(schedule, played_db)
            schedule = _merge_schedule_frames([schedule])

    return _format_schedule(schedule), source


def schedule_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    """Podsumowanie terminarza per liga."""
    if schedule.empty:
        return pd.DataFrame()

    summary = (
        schedule.groupby(["div", "league", "status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ("Rozegrany", "Zaplanowany"):
        if col not in summary.columns:
            summary[col] = 0
    summary["Razem"] = summary["Rozegrany"] + summary["Zaplanowany"]
    return summary.sort_values("league")
