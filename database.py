"""Operacje na bazie SQLite z wynikami meczów."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd

from config import DB_PATH, EXTRA_COLUMNS, MATCH_DB_COLUMNS, SEED_DB_PATH


SQL_TYPE = {
    "div": "TEXT",
    "season": "TEXT",
    "date": "TEXT",
    "home_team": "TEXT",
    "away_team": "TEXT",
    "fthg": "INTEGER",
    "ftag": "INTEGER",
    "ftr": "TEXT",
    "avg_h": "REAL",
    "avg_d": "REAL",
    "avg_a": "REAL",
    "home_shots": "INTEGER",
    "away_shots": "INTEGER",
    "home_sot": "INTEGER",
    "away_sot": "INTEGER",
    "home_corners": "INTEGER",
    "away_corners": "INTEGER",
    "home_fouls": "INTEGER",
    "away_fouls": "INTEGER",
}


def _parse_dates(series: pd.Series) -> pd.Series:
    """Akceptuje daty Y-m-d i Y-m-d H:M:S zapisane w SQLite."""
    try:
        return pd.to_datetime(series, format="mixed", errors="coerce")
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors="coerce")


def _format_date_strings(series: pd.Series) -> pd.Series:
    parsed = _parse_dates(series)
    return parsed.dt.strftime("%Y-%m-%d")


def _normalize_stored_dates(conn: sqlite3.Connection) -> None:
    """Ujednolica zapis dat w bazie (bez czasu)."""
    conn.execute(
        """
        UPDATE matches
        SET date = substr(date, 1, 10)
        WHERE length(date) > 10
        """
    )


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _migrate_schema(conn: sqlite3.Connection) -> None:
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(matches)").fetchall()
    }
    for col in EXTRA_COLUMNS:
        if col not in existing:
            sql_type = SQL_TYPE.get(col, "REAL")
            conn.execute(f"ALTER TABLE matches ADD COLUMN {col} {sql_type}")


def init_db(db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                div TEXT NOT NULL,
                season TEXT NOT NULL,
                date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                fthg INTEGER,
                ftag INTEGER,
                ftr TEXT,
                avg_h REAL,
                avg_d REAL,
                avg_a REAL,
                home_shots INTEGER,
                away_shots INTEGER,
                home_sot INTEGER,
                away_sot INTEGER,
                home_corners INTEGER,
                away_corners INTEGER,
                home_fouls INTEGER,
                away_fouls INTEGER,
                UNIQUE(div, season, date, home_team, away_team)
            )
            """
        )
        _migrate_schema(conn)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_matches_lookup ON matches(div, date)"
        )
        conn.commit()


def _merge_with_existing_row(incoming: pd.Series, existing: pd.Series) -> pd.Series:
    """Uzupełnia wynik z API; nie nadpisuje kursów/statystyk z co.uk."""
    merged = incoming.copy()
    for col in EXTRA_COLUMNS:
        if col in existing.index and pd.isna(merged.get(col)) and pd.notna(existing[col]):
            merged[col] = existing[col]
    for col in ("fthg", "ftag", "ftr"):
        if pd.notna(incoming.get(col)):
            merged[col] = incoming[col]
    return merged


def _apply_existing_odds(records: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return records

    from schedule import _match_key

    lookup: dict[tuple, pd.Series] = {}
    for _, row in existing.iterrows():
        lookup[_match_key(row)] = row

    rows: list[pd.Series] = []
    for _, row in records.iterrows():
        key = _match_key(row)
        if key in lookup:
            rows.append(_merge_with_existing_row(row, lookup[key]))
        else:
            rows.append(row)
    return pd.DataFrame(rows)


def seed_database_available() -> bool:
    return SEED_DB_PATH.exists()


def restore_from_seed(db_path: Path = DB_PATH) -> int:
    """Kopiuje matches_seed.db → matches.db (pełna historia + kursy z co.uk)."""
    import shutil

    if not SEED_DB_PATH.exists():
        raise FileNotFoundError(f"Brak pliku seed: {SEED_DB_PATH}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SEED_DB_PATH, db_path)
    repair_mixed_dates(db_path)
    return len(load_played_matches(db_path))


def upsert_matches(df: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    """Dodaje nowe mecze; istniejące rekordy są aktualizowane."""
    if df.empty:
        return 0

    t0 = time.perf_counter()
    init_db(db_path)
    records = df.copy()

    history = load_played_matches(db_path)
    if not history.empty:
        from schedule import resolve_fixture_teams

        records = resolve_fixture_teams(records, history)
        records = _apply_existing_odds(records, history)

    for col in MATCH_DB_COLUMNS:
        if col not in records.columns:
            records[col] = None
    records = records[MATCH_DB_COLUMNS]
    records["date"] = _format_date_strings(records["date"])

    placeholders = ", ".join(["?"] * len(MATCH_DB_COLUMNS))
    columns_sql = ", ".join(MATCH_DB_COLUMNS)

    values = []
    for row in records.itertuples(index=False):
        row_vals = []
        for col in MATCH_DB_COLUMNS:
            val = getattr(row, col)
            row_vals.append(None if pd.isna(val) else val)
        values.append(tuple(row_vals))

    with get_connection(db_path) as conn:
        before = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        conn.executemany(
            f"""
            INSERT OR REPLACE INTO matches ({columns_sql})
            VALUES ({placeholders})
            """,
            values,
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]

    elapsed = time.perf_counter() - t0
    removed = deduplicate_matches(db_path)
    if removed:
        print(f"  Baza: usunieto {removed} zduplikowanych meczow")
    print(f"  Baza: zapisano {len(values)} rekordow w {elapsed:.1f}s")
    return after - before


def deduplicate_matches(db_path: Path = DB_PATH) -> int:
    """Usuwa duplikaty tego samego meczu (różne nazwy drużyn API vs co.uk)."""
    from schedule import _match_key, _row_priority

    df = load_all_matches(db_path)
    if df.empty:
        return 0

    work = df.copy()
    work["date"] = _parse_dates(work["date"])
    work["_key"] = work.apply(_match_key, axis=1)
    work["_prio"] = work.apply(_row_priority, axis=1)
    work = work.sort_values(["_key", "_prio"])
    deduped = work.drop_duplicates(subset=["_key"], keep="last")
    removed = len(work) - len(deduped)
    if removed == 0:
        return 0

    deduped = deduped.drop(columns=["_key", "_prio", "id"], errors="ignore")
    deduped["date"] = _format_date_strings(deduped["date"])
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM matches")
        deduped.to_sql("matches", conn, if_exists="append", index=False)
        conn.commit()
    return removed


def repair_mixed_dates(db_path: Path = DB_PATH) -> int:
    """Naprawia mieszane formaty dat w bazie (Y-m-d vs Y-m-d H:M:S)."""
    init_db(db_path)
    removed = deduplicate_matches(db_path)
    with get_connection(db_path) as conn:
        _normalize_stored_dates(conn)
        conn.commit()
    return removed


def load_all_matches(db_path: Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with get_connection(db_path) as conn:
        needs_repair = conn.execute(
            "SELECT 1 FROM matches WHERE length(date) > 10 LIMIT 1"
        ).fetchone()
    if needs_repair:
        repair_mixed_dates(db_path)

    with get_connection(db_path) as conn:
        df = pd.read_sql("SELECT * FROM matches ORDER BY date, div", conn)

    if df.empty:
        return df

    df["date"] = _parse_dates(df["date"])
    for col in EXTRA_COLUMNS:
        if col in df.columns:
            if col.startswith("avg_"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_played_matches(db_path: Path = DB_PATH) -> pd.DataFrame:
    df = load_all_matches(db_path)
    if df.empty:
        return df
    return df[df["ftr"].isin(["H", "D", "A"])].copy()
