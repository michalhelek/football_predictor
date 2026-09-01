"""Inżynieria cech i ratingi Elo dla drużyn piłkarskich."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL,
    ELO_K,
    FORM_WINDOW,
    LEAGUES,
)


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def _match_outcome(home_goals: int, away_goals: int) -> tuple[float, float]:
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def _update_elo(
    ratings: dict[tuple[str, str], float],
    div: str,
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> None:
    home = (div, home_team)
    away = (div, away_team)
    home_elo = ratings.get(home, ELO_INITIAL)
    away_elo = ratings.get(away, ELO_INITIAL)
    home_adj = home_elo + ELO_HOME_ADVANTAGE
    exp_home = _expected_score(home_adj, away_elo)
    exp_away = 1.0 - exp_home
    act_home, act_away = _match_outcome(home_goals, away_goals)
    ratings[home] = home_elo + ELO_K * (act_home - exp_home)
    ratings[away] = away_elo + ELO_K * (act_away - exp_away)


def compute_elo_ratings(matches: pd.DataFrame) -> dict[tuple[str, str], float]:
    ratings: dict[tuple[str, str], float] = {}
    played = matches[matches["ftr"].isin(["H", "D", "A"])].sort_values("date")

    for _, row in played.iterrows():
        _update_elo(
            ratings,
            row["div"],
            row["home_team"],
            row["away_team"],
            int(row["fthg"]),
            int(row["ftag"]),
        )

    return ratings


def _safe_float(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _odds_features(avg_h, avg_d, avg_a) -> dict[str, float]:
    h, d, a = _safe_float(avg_h), _safe_float(avg_d), _safe_float(avg_a)
    if not h or not d or not a or h <= 1 or d <= 1 or a <= 1:
        return {
            "implied_prob_h": np.nan,
            "implied_prob_d": np.nan,
            "implied_prob_a": np.nan,
            "odds_home_minus_away": np.nan,
        }

    inv = np.array([1 / h, 1 / d, 1 / a])
    probs = inv / inv.sum()
    return {
        "implied_prob_h": float(probs[0]),
        "implied_prob_d": float(probs[1]),
        "implied_prob_a": float(probs[2]),
        "odds_home_minus_away": float(a - h),
    }


def _empty_stats_form() -> dict[str, float]:
    return {
        "sot_for": np.nan,
        "sot_against": np.nan,
        "shots_for": np.nan,
        "shots_against": np.nan,
        "corners_for": np.nan,
        "corners_against": np.nan,
    }


def _stats_form_from_recent(recent: list[dict]) -> dict[str, float]:
    if not recent:
        return _empty_stats_form()

    keys = ["sot_for", "sot_against", "shots_for", "shots_against", "corners_for", "corners_against"]
    totals = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}

    for entry in recent:
        for key in keys:
            val = entry.get(key)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                totals[key] += float(val)
                counts[key] += 1

    return {
        key: (totals[key] / counts[key] if counts[key] else np.nan)
        for key in keys
    }


def _match_form_entry(row: pd.Series, team: str) -> dict:
    if row["home_team"] == team:
        pts = 3 if row["ftr"] == "H" else (1 if row["ftr"] == "D" else 0)
        entry = {
            "points": pts,
            "gf": row["fthg"],
            "ga": row["ftag"],
            "sot_for": _safe_float(row.get("home_sot")),
            "sot_against": _safe_float(row.get("away_sot")),
            "shots_for": _safe_float(row.get("home_shots")),
            "shots_against": _safe_float(row.get("away_shots")),
            "corners_for": _safe_float(row.get("home_corners")),
            "corners_against": _safe_float(row.get("away_corners")),
        }
    else:
        pts = 3 if row["ftr"] == "A" else (1 if row["ftr"] == "D" else 0)
        entry = {
            "points": pts,
            "gf": row["ftag"],
            "ga": row["fthg"],
            "sot_for": _safe_float(row.get("away_sot")),
            "sot_against": _safe_float(row.get("home_sot")),
            "shots_for": _safe_float(row.get("away_shots")),
            "shots_against": _safe_float(row.get("home_shots")),
            "corners_for": _safe_float(row.get("away_corners")),
            "corners_against": _safe_float(row.get("home_corners")),
        }
    return entry


def _form_from_recent(recent: list[dict]) -> dict[str, float]:
    if not recent:
        return {"points": 1.0, "goals_for": 1.0, "goals_against": 1.0, "matches": 0}
    points = sum(m["points"] for m in recent)
    gf = sum(m["gf"] for m in recent)
    ga = sum(m["ga"] for m in recent)
    n = len(recent)
    return {"points": points / n, "goals_for": gf / n, "goals_against": ga / n, "matches": n}


def _team_form(
    history: pd.DataFrame,
    team: str,
    div: str,
    before_date: pd.Timestamp,
    window: int = FORM_WINDOW,
) -> tuple[dict[str, float], dict[str, float]]:
    past = history[
        (history["date"] < before_date)
        & (history["div"] == div)
        & ((history["home_team"] == team) | (history["away_team"] == team))
    ].tail(window)

    if past.empty:
        return (
            {"points": 1.0, "goals_for": 1.0, "goals_against": 1.0, "matches": 0},
            _empty_stats_form(),
        )

    recent = [_match_form_entry(row, team) for _, row in past.iterrows()]
    return _form_from_recent(recent), _stats_form_from_recent(recent)


def _build_feature_row(
    div,
    date,
    home,
    away,
    ftr,
    home_elo,
    away_elo,
    home_form,
    away_form,
    home_stats,
    away_stats,
    odds_row,
) -> dict:
    odds = _odds_features(
        odds_row.get("avg_h"),
        odds_row.get("avg_d"),
        odds_row.get("avg_a"),
    )
    return {
        "div": div,
        "date": date,
        "home_team": home,
        "away_team": away,
        "ftr": ftr,
        "elo_diff": home_elo - away_elo,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "home_form_pts": home_form["points"],
        "away_form_pts": away_form["points"],
        "home_form_gf": home_form["goals_for"],
        "away_form_gf": away_form["goals_for"],
        "home_form_ga": home_form["goals_against"],
        "away_form_ga": away_form["goals_against"],
        "form_pts_diff": home_form["points"] - away_form["points"],
        "form_gf_diff": home_form["goals_for"] - away_form["goals_for"],
        "form_ga_diff": away_form["goals_against"] - home_form["goals_against"],
        "home_form_sot": home_stats["sot_for"],
        "away_form_sot": away_stats["sot_for"],
        "home_form_sot_against": home_stats["sot_against"],
        "away_form_sot_against": away_stats["sot_against"],
        "home_form_shots": home_stats["shots_for"],
        "away_form_shots": away_stats["shots_for"],
        "home_form_corners": home_stats["corners_for"],
        "away_form_corners": away_stats["corners_for"],
        "form_sot_diff": home_stats["sot_for"] - away_stats["sot_for"],
        "form_sot_against_diff": away_stats["sot_against"] - home_stats["sot_against"],
        **odds,
    }


def build_training_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Buduje macierz cech dla rozegranych meczów (do treningu modelu)."""
    played = matches[matches["ftr"].isin(["H", "D", "A"])].sort_values("date").copy()
    if played.empty:
        return pd.DataFrame()

    ratings: dict[tuple[str, str], float] = {}
    form_history: dict[tuple[str, str], list[dict]] = {}
    rows: list[dict] = []

    for _, row in played.iterrows():
        div = row["div"]
        date = row["date"]
        home = row["home_team"]
        away = row["away_team"]

        home_elo = ratings.get((div, home), ELO_INITIAL)
        away_elo = ratings.get((div, away), ELO_INITIAL)
        home_form = _form_from_recent(form_history.get((div, home), []))
        away_form = _form_from_recent(form_history.get((div, away), []))
        home_stats = _stats_form_from_recent(form_history.get((div, home), []))
        away_stats = _stats_form_from_recent(form_history.get((div, away), []))

        rows.append(
            _build_feature_row(
                div,
                date,
                home,
                away,
                row["ftr"],
                home_elo,
                away_elo,
                home_form,
                away_form,
                home_stats,
                away_stats,
                row,
            )
        )

        _update_elo(ratings, div, home, away, int(row["fthg"]), int(row["ftag"]))

        for team in (home, away):
            key = (div, team)
            entry = _match_form_entry(row, team)
            history = form_history.setdefault(key, [])
            history.append(entry)
            if len(history) > FORM_WINDOW:
                history.pop(0)

    return pd.DataFrame(rows)


def build_prediction_features(
    fixtures: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Buduje cechy dla nadchodzących meczów na podstawie historii."""
    if fixtures.empty:
        return pd.DataFrame()

    ratings = compute_elo_ratings(history[history["ftr"].isin(["H", "D", "A"])])
    rows: list[dict] = []

    for _, row in fixtures.iterrows():
        div = row["div"]
        date = row["date"]
        home = row["home_team"]
        away = row["away_team"]

        home_elo = ratings.get((div, home), ELO_INITIAL)
        away_elo = ratings.get((div, away), ELO_INITIAL)
        home_form, home_stats = _team_form(history, home, div, date)
        away_form, away_stats = _team_form(history, away, div, date)

        rows.append(
            _build_feature_row(
                div,
                date,
                home,
                away,
                None,
                home_elo,
                away_elo,
                home_form,
                away_form,
                home_stats,
                away_stats,
                row,
            )
        )

    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "home_form_pts",
    "away_form_pts",
    "home_form_gf",
    "away_form_gf",
    "home_form_ga",
    "away_form_ga",
    "form_pts_diff",
    "form_gf_diff",
    "form_ga_diff",
    "home_form_sot",
    "away_form_sot",
    "home_form_sot_against",
    "away_form_sot_against",
    "home_form_shots",
    "away_form_shots",
    "home_form_corners",
    "away_form_corners",
    "form_sot_diff",
    "form_sot_against_diff",
    "implied_prob_h",
    "implied_prob_d",
    "implied_prob_a",
    "odds_home_minus_away",
]

LEAGUE_LABELS = {code: idx for idx, code in enumerate(LEAGUES.keys())}


def add_league_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["league_code"] = out["div"].map(LEAGUE_LABELS).fillna(-1).astype(int)
    return out


def compute_elo_timeline(matches: pd.DataFrame) -> pd.DataFrame:
    """Historia Elo drużyn po każdym rozegranym meczu."""
    played = matches[matches["ftr"].isin(["H", "D", "A"])].sort_values("date")
    if played.empty:
        return pd.DataFrame()

    ratings: dict[tuple[str, str], float] = {}
    rows: list[dict] = []

    for _, row in played.iterrows():
        div = row["div"]
        home = row["home_team"]
        away = row["away_team"]
        home_elo = ratings.get((div, home), ELO_INITIAL)
        away_elo = ratings.get((div, away), ELO_INITIAL)

        _update_elo(ratings, div, home, away, int(row["fthg"]), int(row["ftag"]))

        rows.append(
            {
                "date": row["date"],
                "div": div,
                "team": home,
                "opponent": away,
                "venue": "home",
                "elo": ratings[(div, home)],
            }
        )
        rows.append(
            {
                "date": row["date"],
                "div": div,
                "team": away,
                "opponent": home,
                "venue": "away",
                "elo": ratings[(div, away)],
            }
        )

    return pd.DataFrame(rows)


def get_team_recent_matches(
    matches: pd.DataFrame,
    div: str,
    team: str,
    n: int = 10,
) -> pd.DataFrame:
    """Ostatnie N meczów drużyny z wynikiem W/D/L i bramkami."""
    played = matches[matches["ftr"].isin(["H", "D", "A"])].copy()
    team_matches = played[
        (played["div"] == div)
        & ((played["home_team"] == team) | (played["away_team"] == team))
    ].sort_values("date").tail(n)

    if team_matches.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, row in team_matches.iterrows():
        is_home = row["home_team"] == team
        if is_home:
            gf, ga = int(row["fthg"]), int(row["ftag"])
            opponent = row["away_team"]
            result = row["ftr"]
            outcome = "W" if result == "H" else ("D" if result == "D" else "L")
            points = 3 if outcome == "W" else (1 if outcome == "D" else 0)
        else:
            gf, ga = int(row["ftag"]), int(row["fthg"])
            opponent = row["home_team"]
            result = row["ftr"]
            outcome = "W" if result == "A" else ("D" if result == "D" else "L")
            points = 3 if outcome == "W" else (1 if outcome == "D" else 0)

        rows.append(
            {
                "date": row["date"],
                "opponent": opponent,
                "venue": "H" if is_home else "A",
                "score": f"{gf}:{ga}",
                "outcome": outcome,
                "points": points,
            }
        )

    return pd.DataFrame(rows)
