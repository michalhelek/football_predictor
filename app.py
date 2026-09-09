"""Aplikacja webowa Streamlit — prognozy piłkarskie (football-data.co.uk)."""

from __future__ import annotations

import os
import sys
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

# Windows: polskie znaki w terminalu / logach Streamlit
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from compare_models import format_comparison_report
from config import COMPARISON_PATH, DB_PATH, LEAGUES, MANUAL_PATH, PREDICTIONS_PATH
from database import load_played_matches
from features import compute_elo_timeline, get_team_recent_matches
from prediction_history import archive_predictions, history_summary, load_prediction_history
from predictor import run_model_comparison, run_predictions
from schedule import load_season_schedule, schedule_summary
from updater import initialize_database, is_predicted_round_complete, update_after_round

st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
)

LABEL_COLORS = {
    "Wygrana gospodarzy": "#2ecc71",
    "Remis": "#f39c12",
    "Wygrana gości": "#e74c3c",
}
OUTCOME_COLORS = {"W": "#2ecc71", "D": "#f39c12", "L": "#e74c3c"}


def _pct(value: float) -> str:
    return f"{value:.0%}"


def _y_domain(
    values,
    *,
    padding_ratio: float = 0.1,
    floor_zero: bool = False,
    min_top: float | None = None,
) -> tuple[float, float]:
    """Zakres osi Y dopasowany do danych (zamiast skali od 0), żeby wykres nie był płaski."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return 0.0, 1.0

    y_min = float(series.min())
    y_max = float(series.max())
    if y_min == y_max:
        step = abs(y_min) * 0.05 if y_min != 0 else 1.0
        y_min -= step
        y_max += step
    else:
        pad = (y_max - y_min) * padding_ratio
        y_min -= pad
        y_max += pad

    if floor_zero and y_min > 0:
        y_min = 0.0
    if min_top is not None:
        y_max = max(y_max, min_top)

    return y_min, y_max


def _render_scaled_line_chart(df: pd.DataFrame, *, height: int = 280, y_label: str | None = None) -> None:
    plot = df.reset_index()
    x_col, y_col = plot.columns[0], plot.columns[1]
    y_min, y_max = _y_domain(plot[y_col], padding_ratio=0.08)

    chart = (
        alt.Chart(plot)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{x_col}:T", title="Data"),
            y=alt.Y(
                f"{y_col}:Q",
                scale=alt.Scale(domain=[y_min, y_max]),
                title=y_label or y_col,
            ),
            tooltip=[alt.Tooltip(f"{x_col}:T", title="Data"), alt.Tooltip(f"{y_col}:Q", format=".1f")],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_scaled_bar_chart(
    df: pd.DataFrame,
    *,
    height: int = 280,
    y_label: str | None = None,
    floor_zero: bool = True,
    categorical_x: bool = False,
    min_top: float | None = None,
) -> None:
    plot = df.reset_index()
    x_col, y_col = plot.columns[0], plot.columns[1]
    y_min, y_max = _y_domain(
        plot[y_col],
        padding_ratio=0.12,
        floor_zero=floor_zero,
        min_top=min_top,
    )
    x_type = "N" if categorical_x else "T"
    x_title = "" if categorical_x else "Data"

    chart = (
        alt.Chart(plot)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:{x_type}", title=x_title),
            y=alt.Y(
                f"{y_col}:Q",
                scale=alt.Scale(domain=[y_min, y_max]),
                title=y_label or y_col,
            ),
            tooltip=[
                alt.Tooltip(f"{x_col}:{x_type}", title="Data" if not categorical_x else "Wynik"),
                alt.Tooltip(f"{y_col}:Q", format=".2f"),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, use_container_width=True)


@st.cache_data(ttl=300)
def _load_played() -> pd.DataFrame:
    return load_played_matches()


@st.cache_data(ttl=300)
def _load_elo_timeline() -> pd.DataFrame:
    return compute_elo_timeline(_load_played())


@st.cache_data(ttl=300)
def _load_status() -> dict:
    played = _load_played()
    status = {
        "total": len(played),
        "played": len(played),
        "with_odds": int(played["avg_h"].notna().sum()) if not played.empty else 0,
        "with_sot": int(played["home_sot"].notna().sum()) if not played.empty else 0,
        "latest_match": (
            played["date"].max().strftime("%Y-%m-%d") if not played.empty else "—"
        ),
        "db_exists": DB_PATH.exists(),
    }
    if PREDICTIONS_PATH.exists():
        preds = pd.read_csv(PREDICTIONS_PATH, encoding="utf-8-sig")
        status["predictions_count"] = len(preds)
        status["round_complete"] = is_predicted_round_complete(preds, played)
    else:
        status["predictions_count"] = 0
        status["round_complete"] = False
    return status


def _clear_caches() -> None:
    _load_played.clear()
    _load_elo_timeline.clear()
    _load_status.clear()
    _load_schedule.clear()
    _load_predictions_from_disk.clear()


def _style_predictions(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in ("prob_H", "prob_D", "prob_A", "confidence"):
        if col in display.columns:
            display[col] = display[col].map(_pct)
    return display


def _filter_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    min_date = work["date"].min().date()
    max_date = work["date"].max().date()

    c1, c2, c3 = st.columns(3)
    with c1:
        leagues = ["Wszystkie"] + sorted(work["league"].dropna().unique().tolist())
        default_idx = 0
        if "pred_league" in st.session_state and st.session_state["pred_league"] in leagues:
            default_idx = leagues.index(st.session_state["pred_league"])
        league = st.selectbox("Liga", leagues, index=default_idx, key="pred_league")
    with c2:
        date_from = st.date_input("Od", min_date, min_value=min_date, max_value=max_date, key="pred_from")
    with c3:
        date_to = st.date_input("Do", max_date, min_value=min_date, max_value=max_date, key="pred_to")

    if league != "Wszystkie":
        work = work[work["league"] == league]
    work = work[(work["date"].dt.date >= date_from) & (work["date"].dt.date <= date_to)]
    return work.sort_values("date")


def _prob_chart(row: pd.Series) -> None:
    probs = pd.DataFrame(
        {
            "Wynik": ["H (gosp.)", "Remis", "A (goście)"],
            "Prawdopodobieństwo": [row["prob_H"], row["prob_D"], row["prob_A"]],
        }
    ).set_index("Wynik")
    _render_scaled_bar_chart(
        probs,
        height=140,
        y_label="P",
        floor_zero=False,
        categorical_x=True,
    )


def _predictions_file_mtime() -> float:
    if PREDICTIONS_PATH.exists():
        return PREDICTIONS_PATH.stat().st_mtime
    return 0.0


@st.cache_data(ttl=15)
def _load_predictions_from_disk(file_mtime: float) -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PREDICTIONS_PATH, encoding="utf-8-sig")


def _predictions_summary(df: pd.DataFrame) -> None:
    if df.empty or "league" not in df.columns:
        return
    summary = (
        df.groupby("league")
        .size()
        .reset_index(name="mecze")
        .sort_values("league")
    )
    st.dataframe(summary, hide_index=True, use_container_width=True)


def _render_predictions(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Brak prognoz. Kliknij **Generuj prognozy** w panelu bocznym.")
        return

    leagues_n = df["league"].nunique() if "league" in df.columns else 0
    if leagues_n < 6:
        model_used = df["model"].iloc[0] if "model" in df.columns and not df.empty else "?"
        if model_used == "odds":
            st.warning(
                f"W pliku jest tylko **{leagues_n} lig** ({len(df)} meczów). "
                "Model **Kursy** działa tylko tam, gdzie są kursy bukmacherskie w terminarzu. "
                "Wybierz **Auto**, **Ensemble** lub **XGBoost** i kliknij **Generuj prognozy**."
            )
        else:
            st.warning(
                f"W pliku jest tylko **{leagues_n} lig** ({len(df)} meczów). "
                "Kliknij **Generuj prognozy** (potrzebny token API w `data/.api_token`, ok. 1 min)."
            )

    st.subheader(f"Najbliższa kolejka — {leagues_n} lig, {len(df)} meczów")
    st.caption(
        "Prognozy najbliższych meczów w każdej lidze (w tym nakładające się kolejki, "
        "np. La Liga). Pełny terminarz sezonu jest w zakładce **Terminarz**."
    )
    _predictions_summary(df)

    filtered = _filter_predictions(df)
    if filtered.empty:
        st.warning("Brak prognoz dla wybranych filtrów.")
        return

    st.caption(f"{len(filtered)} meczów | model: {filtered['model'].iloc[0]}")

    for _, row in filtered.iterrows():
        color = LABEL_COLORS.get(row["prediction_label"], "#3498db")
        round_label = ""
        if "round_id" in row and pd.notna(row["round_id"]):
            round_label = f" · Kolejka {int(row['round_id'])}"
        with st.container(border=True):
            cols = st.columns([3, 2, 2])
            with cols[0]:
                st.markdown(f"**{row['home_team']}** vs **{row['away_team']}**")
                st.caption(
                    f"{row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}"
                    f"{round_label} · {row['league']}"
                )
            with cols[1]:
                st.markdown(
                    f"<span style='color:{color}; font-weight:600;'>"
                    f"{row['prediction_label']}</span>",
                    unsafe_allow_html=True,
                )
                st.progress(
                    float(row["confidence"]),
                    text=f"Pewność {_pct(row['confidence'])}",
                )
            with cols[2]:
                _prob_chart(row)


def _render_team_analysis(played: pd.DataFrame, elo_timeline: pd.DataFrame) -> None:
    if played.empty:
        st.info("Baza jest pusta. Uruchom inicjalizację w panelu bocznym.")
        return

    div_options = list(LEAGUES.keys())
    c1, c2 = st.columns(2)
    with c1:
        div = st.selectbox(
            "Liga",
            div_options,
            format_func=lambda code: LEAGUES[code],
            key="team_div",
        )
    teams = sorted(
        set(played.loc[played["div"] == div, "home_team"])
        | set(played.loc[played["div"] == div, "away_team"])
    )
    with c2:
        team = st.selectbox("Drużyna", teams, key="team_name")

    team_elo = elo_timeline[
        (elo_timeline["div"] == div) & (elo_timeline["team"] == team)
    ].copy()
    recent = get_team_recent_matches(played, div, team, n=10)

    m1, m2, m3 = st.columns(3)
    current_elo = float(team_elo["elo"].iloc[-1]) if not team_elo.empty else 1500.0
    m1.metric("Aktualne Elo", f"{current_elo:.0f}")
    if not recent.empty:
        m2.metric("Forma (pkt/mecz)", f"{recent['points'].mean():.1f}")
        m3.metric("Bramki (śr.)", f"{recent['score'].str.split(':').str[0].astype(int).mean():.1f}")
    else:
        m2.metric("Forma (pkt/mecz)", "—")
        m3.metric("Bramki (śr.)", "—")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Wykres Elo")
        if team_elo.empty:
            st.caption("Brak danych Elo dla tej drużyny.")
        else:
            chart_df = team_elo.set_index("date")[["elo"]]
            _render_scaled_line_chart(chart_df, height=280, y_label="Elo")

    with chart_cols[1]:
        st.subheader("Ostatnie mecze")
        if recent.empty:
            st.caption("Brak rozegranych meczów.")
        else:
            form_df = recent.set_index("date")[["points"]]
            _render_scaled_bar_chart(
                form_df,
                height=280,
                y_label="Punkty",
                floor_zero=True,
                min_top=3.0,
            )
            display = recent.copy()
            display["date"] = display["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(display, hide_index=True, use_container_width=True)


def _render_prediction_history(played: pd.DataFrame) -> None:
    history = load_prediction_history(played)
    if history.empty:
        if PREDICTIONS_PATH.exists():
            current = pd.read_csv(PREDICTIONS_PATH, encoding="utf-8-sig")
            if not current.empty:
                archive_predictions(current)
                history = load_prediction_history(played)

    if history.empty:
        st.info(
            "Brak archiwum prognoz. Wygeneruj prognozy — każda kolejka "
            "będzie zapisywana automatycznie."
        )
        return

    summary = history_summary(history)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prognozy w archiwum", summary["total_predictions"])
    c2.metric("Ocenione mecze", summary["evaluated"])
    c3.metric("Trafione", summary["correct"])
    c4.metric(
        "Dokładność",
        f"{summary['accuracy']:.1%}" if summary["accuracy"] is not None else "—",
    )

    if not summary["by_model"].empty:
        st.subheader("Trafność wg modelu")
        model_view = summary["by_model"].copy()
        model_view["accuracy"] = model_view["accuracy"].map(_pct)
        st.dataframe(model_view, hide_index=True, use_container_width=True)

    if not summary["by_league"].empty:
        st.subheader("Trafność wg ligi")
        league_view = summary["by_league"].copy()
        _render_scaled_bar_chart(
            league_view.set_index("league")[["accuracy"]],
            height=220,
            y_label="Dokładność",
            floor_zero=False,
            categorical_x=True,
        )
        league_view["accuracy"] = league_view["accuracy"].map(_pct)
        st.dataframe(league_view, hide_index=True, use_container_width=True)

    completed = history[history["evaluated"]].copy()
    if not completed.empty:
        completed = completed.sort_values("date", ascending=False)
        completed["date"] = completed["date"].dt.strftime("%Y-%m-%d")
        completed["correct_label"] = completed["correct"].map({True: "✓", False: "✗"})
        show_cols = [
            "date",
            "league",
            "home_team",
            "away_team",
            "predicted",
            "actual_ftr",
            "correct_label",
            "model",
            "confidence",
        ]
        st.subheader("Historia ocenionych prognoz")
        st.dataframe(
            completed[show_cols].rename(columns={"correct_label": "trafione"}),
            hide_index=True,
            use_container_width=True,
        )


@st.cache_data(ttl=3600)
def _load_schedule() -> tuple[pd.DataFrame, str]:
    return load_season_schedule(played_db=_load_played())


def _render_schedule() -> None:
    from data_loader import _current_season_code

    season = _current_season_code()
    st.caption(
        f"Sezon {season[:2]}/{season[2:]} · pełny terminarz z football-data.org "
        "(co.uk ma tylko najbliższe mecze z kursami)"
    )

    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        if st.button("Odśwież terminarz", key="refresh_schedule"):
            _load_schedule.clear()
            st.rerun()

    with st.spinner("Pobieranie terminarza (6 lig, ok. 1 min)..."):
        schedule, source = _load_schedule()

    st.info(f"Źródło danych: **{source}**")

    if schedule.empty:
        st.warning("Brak danych terminarza. Sprawdź połączenie z internetem.")
        return

    summary = schedule_summary(schedule)
    st.dataframe(summary, hide_index=True, use_container_width=True)

    work = schedule.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    min_date = work["date"].min().date()
    max_date = work["date"].max().date()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        league_options = ["Wszystkie"] + sorted(work["league"].unique().tolist())
        league = st.selectbox("Liga", league_options, key="sched_league")
    with c2:
        status_filter = st.selectbox(
            "Status",
            ["Wszystkie", "Zaplanowany", "Rozegrany"],
            key="sched_status",
        )
    with c3:
        date_from = st.date_input(
            "Od",
            min_date,
            min_value=min_date,
            max_value=max_date,
            key="sched_from",
        )
    with c4:
        date_to = st.date_input(
            "Do",
            max_date,
            min_value=min_date,
            max_value=max_date,
            key="sched_to",
        )

    if date_from > date_to:
        date_from, date_to = date_to, date_from

    league_scope = work if league == "Wszystkie" else work[work["league"] == league]
    rounds = sorted(league_scope["round_id"].dropna().unique().tolist())
    round_options = ["Wszystkie"] + [f"Kolejka {int(r)}" for r in rounds]
    round_pick = st.selectbox(
        "Kolejka",
        round_options,
        key="sched_round",
        help=(
            "Numer kolejki liczony osobno w każdej lidze. "
            "Aby zobaczyć mecze z konkretnych dat, użyj filtrów Od/Do."
        ),
    )

    filtered = work.copy()
    if league != "Wszystkie":
        filtered = filtered[filtered["league"] == league]
    if status_filter != "Wszystkie":
        filtered = filtered[filtered["status"] == status_filter]
    filtered = filtered[
        (filtered["date"].dt.date >= date_from) & (filtered["date"].dt.date <= date_to)
    ]
    if round_pick != "Wszystkie":
        round_num = int(round_pick.replace("Kolejka ", ""))
        filtered = filtered[filtered["round_id"] == round_num]

    display_cols = [
        "round_id",
        "date",
        "league",
        "home_team",
        "away_team",
        "score",
        "result_label",
        "status",
        "avg_h",
        "avg_d",
        "avg_a",
    ]
    view = filtered[display_cols].copy().sort_values(["date", "league", "round_id"])
    view["date"] = view["date"].dt.strftime("%Y-%m-%d")
    view = view.rename(
        columns={
            "round_id": "Kolejka",
            "date": "Data",
            "league": "Liga",
            "home_team": "Gospodarze",
            "away_team": "Goście",
            "score": "Wynik",
            "result_label": "H/D/A",
            "status": "Status",
            "avg_h": "Kurs H",
            "avg_d": "Kurs D",
            "avg_a": "Kurs A",
        }
    )

    st.caption(
        f"{len(view)} meczów w przedziale {date_from} – {date_to} "
        f"(łącznie w terminarzu: {len(work)})"
    )
    st.dataframe(view, hide_index=True, use_container_width=True, height=min(700, 35 * len(view) + 38))

    st.download_button(
        "Pobierz terminarz CSV",
        filtered.to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"terminarz_{season}.csv",
        mime="text/csv",
        key="download_schedule",
    )


@st.cache_data(ttl=60)
def _load_manual() -> str:
    if not MANUAL_PATH.exists():
        return "Brak pliku INSTRUKCJA.md w folderze projektu."
    return MANUAL_PATH.read_text(encoding="utf-8")


def _render_manual() -> None:
    content = _load_manual()
    st.download_button(
        "Pobierz instrukcję (Markdown)",
        content,
        file_name="INSTRUKCJA.md",
        mime="text/markdown",
    )
    st.markdown(content, unsafe_allow_html=False)


def main() -> None:
    st.title("⚽ Football Predictor")
    st.markdown(
        "Prognozy wyników H/D/A dla 6 lig europejskich — dane z "
        "[football-data.co.uk](https://www.football-data.co.uk)."
    )

    played = _load_played()
    elo_timeline = _load_elo_timeline()
    status = _load_status()

    with st.sidebar:
        st.header("Panel sterowania")
        model = st.selectbox(
            "Model",
            ["auto", "dixon_coles", "xgboost", "odds", "ensemble"],
            format_func=lambda x: {
                "auto": "Auto (lepszy z porównania)",
                "dixon_coles": "Dixon-Coles",
                "xgboost": "XGBoost",
                "odds": "Kursy bukmacherskie",
                "ensemble": "Ensemble (DC + XGBoost)",
            }[x],
        )

        st.divider()
        if st.button("🚀 Inicjalizacja bazy", use_container_width=True):
            try:
                with st.spinner(
                    "Ładowanie bazy (seed + kursy lub pobieranie z internetu, ok. 1–5 min)..."
                ):
                    result = initialize_database(save_raw=False)
                    _clear_caches()
                if result.get("source") == "seed":
                    st.success(
                        f"Załadowano **{result['added']}** meczów z bazy seed "
                        f"(**{result['with_odds']}** z kursami). "
                        f"Aktualizacje z API: {result.get('api_updates', 0)}."
                    )
                else:
                    st.success(
                        f"Zapisano **{result['added']}** meczów "
                        f"(**{result['with_odds']}** z kursami). "
                        "Uwaga: samo API nie ma kursów — dodaj matches_seed.db do repo."
                    )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
                try:
                    from api_loader import validate_api_token
                    from config import get_data_source

                    ok, msg = validate_api_token()
                    st.caption(
                        f"Źródło: **{get_data_source()}** | "
                        f"test API: **{'OK' if ok else 'BŁĄD'}** — {msg}"
                    )
                except Exception as token_exc:
                    st.caption(f"Test API: **BŁĄD** ({token_exc})")
                st.info(
                    "1. Wejdź na https://www.football-data.org/client/register → **Account** → skopiuj **API Token**\n\n"
                    "2. W **Settings → Secrets** wklej (nowy token, jeśli stary nie działa):\n\n"
                    "```toml\n"
                    "FOOTBALL_DATA_ORG_TOKEN = \"wklej_token_z_konta\"\n"
                    "FOOTBALL_DATA_SOURCE = \"api\"\n"
                    "```\n\n"
                    "3. **Save** → poczekaj 3–5 min na **Inicjalizacja bazy**"
                )

        if st.button("📊 Generuj prognozy", use_container_width=True):
            try:
                with st.spinner("Pobieranie terminarza i obliczanie prognoz (ok. 1 min)..."):
                    preds = run_predictions(model_name=model)
                _clear_caches()
                if preds.empty:
                    st.error(
                        "Brak meczów do prognozy. Sprawdź token API (`data/.api_token`) "
                        "i połączenie z internetem."
                    )
                else:
                    n_lig = preds["league"].nunique() if "league" in preds.columns else 0
                    for key in ("pred_league", "pred_from", "pred_to"):
                        st.session_state.pop(key, None)
                    st.session_state["pred_league"] = "Wszystkie"
                    st.success(f"Zapisano {len(preds)} prognoz w {n_lig} ligach.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Błąd generowania prognoz: {exc}")
                st.exception(exc)

        if st.button("⚖️ Porównaj modele", use_container_width=True):
            with st.spinner("Trening i ewaluacja (1–3 min)..."):
                result = run_model_comparison()
                st.session_state["comparison"] = result
            st.success(f"Lepszy model: **{result['winner']}**")

        if st.button("🔄 Odśwież dane", use_container_width=True):
            from data_loader import download_historical_data
            from database import upsert_matches

            with st.spinner("Pobieranie CSV z football-data.co.uk..."):
                data = download_historical_data(save_raw=True, source="csv")
                updated = upsert_matches(data)
                _clear_caches()
            if updated:
                st.success(f"Zaktualizowano {updated} rekordów.")
            else:
                st.warning(
                    "Brak nowych danych z co.uk (serwis może być chwilowo niedostępny). "
                    "Na Streamlit Cloud kursy masz już w bazie seed."
                )

        force_update = st.checkbox("Wymuś aktualizację (--force)")
        if st.button("📥 Aktualizuj po kolejce", use_container_width=True):
            try:
                with st.spinner("Pobieranie wyników i ponowny trening (ok. 2–4 min)..."):
                    result = update_after_round(force=force_update)
                    _clear_caches()
                if result["updated"]:
                    n_preds = result.get("new_predictions_count", 0)
                    st.success(
                        f"Baza i modele zaktualizowane. "
                        f"Nowe prognozy: {n_preds} meczów."
                    )
                    _clear_caches()
                    st.rerun()
                else:
                    st.warning(result["reason"])
            except Exception as exc:
                from api_loader import FootballDataOrgError

                _clear_caches()
                if isinstance(exc, FootballDataOrgError) or "429" in str(exc):
                    st.error(str(exc))
                    st.info(
                        "Limit darmowego API (~10 zapytań/min). "
                        "Poczekaj **1–2 minuty** i kliknij ponownie. "
                        "Przy dużej bazie (seed) pełna historia nie jest już pobierana — "
                        "aktualizowany jest tylko bieżący sezon."
                    )
                else:
                    st.error(f"Błąd aktualizacji: {exc}")
                    st.exception(exc)

        st.divider()
        pred_mtime = _predictions_file_mtime()
        if pred_mtime:
            preds_info = _load_predictions_from_disk(pred_mtime)
            n_lig = preds_info["league"].nunique() if not preds_info.empty and "league" in preds_info.columns else 0
            ts = datetime.fromtimestamp(pred_mtime).strftime("%Y-%m-%d %H:%M")
            st.caption(f"Plik prognoz: {len(preds_info)} mecz., {n_lig} lig")
            st.caption(f"Ostatnia aktualizacja: {ts}")
        st.metric("Mecze w bazie", status["played"])
        st.caption(f"Ostatni wynik w bazie: {status['latest_match']}")
        st.metric("Z kursami", status["with_odds"])
        if status["predictions_count"]:
            label = "TAK" if status["round_complete"] else "NIE"
            st.metric("Kolejka rozegrana", label)

    tabs = st.tabs(
        [
            "Prognozy",
            "Terminarz",
            "Analiza drużyn",
            "Historia prognoz",
            "Porównanie modeli",
            "Tabela prognoz",
            "Instrukcja",
        ]
    )

    preds = _load_predictions_from_disk(_predictions_file_mtime())

    with tabs[0]:
        _render_predictions(preds)

    with tabs[1]:
        _render_schedule()

    with tabs[2]:
        _render_team_analysis(played, elo_timeline)

    with tabs[3]:
        _render_prediction_history(played)

    with tabs[4]:
        if "comparison" in st.session_state:
            result = st.session_state["comparison"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Dixon-Coles", f"{result['dixon_coles']['accuracy']:.1%}")
            c2.metric("XGBoost", f"{result['xgboost']['accuracy']:.1%}")
            c3.metric("MLP", f"{result['mlp']['accuracy']:.1%}")
            c4.metric("Kursy", f"{result['odds']['accuracy']:.1%}")
            c5.metric("Ensemble", f"{result['ensemble']['accuracy']:.1%}")
            st.caption(f"Lepszy model: **{result['winner']}**")
            st.text(format_comparison_report(result))
        elif COMPARISON_PATH.exists():
            st.text(COMPARISON_PATH.read_text(encoding="utf-8"))
        else:
            st.info("Uruchom **Porównaj modele** w panelu bocznym.")

    with tabs[5]:
        preds = _load_predictions_from_disk(_predictions_file_mtime())
        if not preds.empty:
            st.dataframe(
                _style_predictions(preds),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Pobierz CSV",
                preds.to_csv(index=False, encoding="utf-8-sig"),
                file_name="predictions.csv",
                mime="text/csv",
            )
        else:
            st.info("Brak prognoz do wyświetlenia.")

    with tabs[6]:
        _render_manual()


if __name__ == "__main__":
    main()
