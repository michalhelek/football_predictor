#!/usr/bin/env python3
"""
System prognozowania wyników piłkarskich.

Użycie:
    python main.py init          – pobierz 5 lat historii i wytrenuj model
    python main.py predict       – wygeneruj prognozy najbliższej kolejki
    python main.py refresh        – odśwież kursy i statystyki (CSV)
    python main.py set-token TOKEN – zapisz token football-data.org
    python main.py compare        – porównaj Dixon-Coles vs XGBoost
    python main.py update        – dopisz wyniki i przelicz prognozy
    python main.py update --force – wymuś aktualizację bez czekania na wyniki
    python main.py status        – pokaż stan bazy i ostatnie prognozy
    python main.py schedule      – pokaż terminarz bieżącego sezonu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Umożliwia uruchomienie bez instalacji pakietu
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import API_TOKEN_FILE, LEAGUES, PREDICTIONS_PATH
from database import load_all_matches, load_played_matches
from features import build_training_features
from model import train_model
from predictor import run_model_comparison, run_predictions
from updater import initialize_database, is_predicted_round_complete, update_after_round


def cmd_init(_: argparse.Namespace) -> None:
    print("Pobieranie danych historycznych (6 lig x 5 sezonow + kursy + statystyki)...")
    added = initialize_database()
    print(f"Zapisano/zaktualizowano mecze w bazie (+{added} nowych rekordów).")

    print("Trening modeli prognozujących (XGBoost + Dixon-Coles)...")
    history = load_played_matches()
    training = build_training_features(history)
    _, xgb_metrics = train_model(training)
    print(f"XGBoost - dokladnosc testowa: {xgb_metrics['accuracy']:.1%}")

    from dixon_coles import train_dixon_coles

    _, dc_info = train_dixon_coles(history)
    print(f"Dixon-Coles - dopasowane ligi: {dc_info['leagues_fitted']}")

    print("\nPorownanie modeli...")
    comparison = run_model_comparison()
    print(comparison["summary"])

    print("\nGenerowanie prognoz najbliższej kolejki...")
    predictions = run_predictions()
    if predictions.empty:
        print("Brak nadchodzących meczów do prognozy.")
    else:
        print(f"Zapisano {len(predictions)} prognoz -> {PREDICTIONS_PATH}")
        _print_predictions(predictions)


def cmd_predict(args: argparse.Namespace) -> None:
    predictions = run_predictions(model_name=args.model)
    if predictions.empty:
        print("Brak nadchodzących meczów do prognozy.")
        return
    print(f"Zapisano {len(predictions)} prognoz -> {PREDICTIONS_PATH}")
    _print_predictions(predictions)


def cmd_set_token(args: argparse.Namespace) -> None:
    token = args.token
    if not token:
        token = input("Wklej token football-data.org: ").strip()
    if not token:
        print("Token nie moze byc pusty.")
        return

    API_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    API_TOKEN_FILE.write_text(token, encoding="utf-8")
    print(f"Token zapisany w: {API_TOKEN_FILE}")
    print("Ustaw zrodlo danych: set FOOTBALL_DATA_SOURCE=api")
    print("  PowerShell: $env:FOOTBALL_DATA_SOURCE='api'")


def cmd_refresh(_: argparse.Namespace) -> None:
    from data_loader import download_historical_data
    from database import upsert_matches

    print("Odswiezanie bazy (kursy + statystyki meczow)...")
    data = download_historical_data(save_raw=True)
    updated = upsert_matches(data)
    print(f"Zaktualizowano rekordow: {updated}")
    print(f"Mecze z kursami: {data['avg_h'].notna().sum()} / {len(data)}")
    print(f"Mecze ze strzalami celnymi: {data['home_sot'].notna().sum()} / {len(data)}")


def cmd_compare(_: argparse.Namespace) -> None:
    from compare_models import format_comparison_report

    result = run_model_comparison()
    print(format_comparison_report(result))


def cmd_update(args: argparse.Namespace) -> None:
    result = update_after_round(force=args.force)
    if not result["updated"]:
        print(result["reason"])
        return

    print("Baza zaktualizowana. Model wytrenowany ponownie.")
    print(f"  Nowe mecze (bieżący sezon): {result['added_current_season']}")
    print(f"  Nowe mecze (historia):       {result['added_historical']}")
    print(f"  XGBoost - trening:           {result['metrics'].get('xgboost', {})}")
    print(f"  Dixon-Coles - trening:       {result['metrics'].get('dixon_coles', {})}")
    print(f"  Nowe prognozy:               {result['new_predictions_count']}")
    print(f"  Plik prognoz:                {result['predictions_path']}")

    preds = run_predictions()
    if not preds.empty:
        _print_predictions(preds)


def cmd_status(_: argparse.Namespace) -> None:
    all_matches = load_all_matches()
    played = load_played_matches()

    print("=== Status systemu ===")
    print(f"Ligi: {', '.join(LEAGUES.values())}")
    print(f"Mecze w bazie:     {len(all_matches)}")
    print(f"Rozegrane mecze:   {len(played)}")

    if PREDICTIONS_PATH.exists():
        import pandas as pd

        preds = pd.read_csv(PREDICTIONS_PATH, encoding="utf-8-sig")
        complete = is_predicted_round_complete(preds)
        print(f"Aktywne prognozy:  {len(preds)} meczów")
        print(f"Kolejka rozegrana: {'TAK' if complete else 'NIE'}")
        if not complete:
            print("  -> Uruchom 'python main.py update' po zakończeniu kolejki.")
    else:
        print("Brak zapisanych prognoz. Uruchom 'python main.py init' lub 'predict'.")


def cmd_schedule(args: argparse.Namespace) -> None:
    from data_loader import _current_season_code
    from schedule import load_season_schedule

    played = load_played_matches()
    schedule, source = load_season_schedule(played_db=played)

    if schedule.empty:
        print("Brak danych terminarza.")
        return

    season = _current_season_code()
    print(f"=== Terminarz sezonu {season[:2]}/{season[2:]} ===")
    print(f"Zrodlo: {source}\n")

    view = schedule
    if args.league:
        view = view[view["div"] == args.league.upper()]
    if args.upcoming:
        view = view[view["status"] == "Zaplanowany"]

    for div in view["div"].unique():
        league_name = LEAGUES.get(div, div)
        league_matches = view[view["div"] == div]
        played_n = (league_matches["status"] == "Rozegrany").sum()
        upcoming_n = (league_matches["status"] == "Zaplanowany").sum()
        print(f"--- {league_name} ({played_n} rozegranych, {upcoming_n} zaplanowanych) ---")
        for _, row in league_matches.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d")
            line = (
                f"  K{row['round_id']:>2} | {date_str} | "
                f"{row['home_team']} vs {row['away_team']}"
            )
            if row["status"] == "Rozegrany":
                line += f" | {row['score']} ({row['result_label']})"
            else:
                odds = ""
                if pd.notna(row.get("avg_h")):
                    odds = f" | kursy H:{row['avg_h']:.2f} D:{row['avg_d']:.2f} A:{row['avg_a']:.2f}"
                line += f" | zaplanowany{odds}"
            print(line)
        print()


def _print_predictions(predictions) -> None:
    print("\n=== Prognozy najblizszej kolejki ===\n")
    for _, row in predictions.iterrows():
        line = (
            f"{row['date']} | {row['league']}\n"
            f"  {row['home_team']} vs {row['away_team']}\n"
            f"  Prognoza: {row['prediction_label']} "
            f"(H:{row['prob_H']:.0%} D:{row['prob_D']:.0%} A:{row['prob_A']:.0%})\n"
        )
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prognozowanie wyników meczów piłkarskich (football-data.co.uk)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Pobierz historię, wytrenuj modele, wygeneruj prognozy")
    p_predict = sub.add_parser("predict", help="Wygeneruj prognozy najbliższej kolejki")
    p_predict.add_argument(
        "--model",
        choices=["auto", "dixon_coles", "xgboost", "odds", "ensemble"],
        default="auto",
        help="Model prognoz (auto = lepszy z ostatniego porownania)",
    )
    p_set_token = sub.add_parser("set-token", help="Zapisz token API football-data.org")
    p_set_token.add_argument("token", nargs="?", help="Token z https://www.football-data.org/client/register")

    sub.add_parser("refresh", help="Pobierz ponownie dane z kursami i statystykami")
    sub.add_parser("compare", help="Porownaj DC, XGBoost, kursy i ensemble")
    sub.add_parser("status", help="Pokaż stan bazy i prognoz")
    p_schedule = sub.add_parser("schedule", help="Terminarz bieżącego sezonu")
    p_schedule.add_argument(
        "--league",
        choices=list(LEAGUES.keys()),
        help="Filtruj po kodzie ligi (np. E0, SP1)",
    )
    p_schedule.add_argument(
        "--upcoming",
        action="store_true",
        help="Pokaż tylko nadchodzące mecze",
    )

    p_update = sub.add_parser("update", help="Aktualizuj dane po rozegraniu kolejki")
    p_update.add_argument(
        "--force",
        action="store_true",
        help="Wymuś aktualizację nawet jeśli kolejka nie jest kompletna",
    )

    args = parser.parse_args()
    commands = {
        "init": cmd_init,
        "predict": cmd_predict,
        "refresh": cmd_refresh,
        "set-token": cmd_set_token,
        "compare": cmd_compare,
        "update": cmd_update,
        "status": cmd_status,
        "schedule": cmd_schedule,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
