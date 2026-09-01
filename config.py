"""Konfiguracja systemu prognozowania wyników piłkarskich."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = DATA_DIR / "matches.db"
MODEL_PATH = DATA_DIR / "model.joblib"
MLP_MODEL_PATH = DATA_DIR / "mlp_model.joblib"
DC_MODEL_PATH = DATA_DIR / "dixon_coles.joblib"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"
PREDICTION_HISTORY_PATH = OUTPUT_DIR / "prediction_history.csv"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison.txt"
MANUAL_PATH = BASE_DIR / "INSTRUKCJA.md"
METADATA_PATH = DATA_DIR / "metadata.json"
API_TOKEN_FILE = DATA_DIR / ".api_token"

BASE_URL = "https://www.football-data.co.uk/mmz4281"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
FOOTBALL_DATA_ORG_API = "https://api.football-data.org/v4"

# Źródło danych: "csv" | "api" | "hybrid"
# csv    = football-data.co.uk (domyślnie, ma kursy i statystyki)
# api    = football-data.org (wymaga tokenu)
# hybrid = historia CSV + fixtures/aktualizacje z API
DATA_SOURCE = os.getenv("FOOTBALL_DATA_SOURCE", "csv").lower()

# Mapowanie kodów CSV (co.uk) -> kody API (football-data.org)
API_LEAGUE_CODES = {
    "E0": "PL",    # Premier League
    "SP1": "PD",   # La Liga
    "D1": "BL1",   # Bundesliga
    "I1": "SA",    # Serie A
    "F1": "FL1",   # Ligue 1
    "N1": "DED",   # Eredivisie
}

# Opóźnienie między zapytaniami API (limit darmowego planu ~10/min)
API_REQUEST_DELAY_SEC = 6.5

# 6 największych lig europejskich (kody football-data.co.uk)
LEAGUES = {
    "E0": "Premier League (Anglia)",
    "SP1": "La Liga (Hiszpania)",
    "D1": "Bundesliga (Niemcy)",
    "I1": "Serie A (Włochy)",
    "F1": "Ligue 1 (Francja)",
    "N1": "Eredivisie (Holandia)",
}

NUM_SEASONS = 5

# Forma drużyny – liczba ostatnich meczów
FORM_WINDOW = 5

# Elo – parametry startowe
ELO_INITIAL = 1500
ELO_K = 32
ELO_HOME_ADVANTAGE = 65

# Kolumny dodatkowe z football-data.co.uk (kursy + statystyki meczu)
ODDS_COLUMNS = ["avg_h", "avg_d", "avg_a"]
STAT_COLUMNS = [
    "home_shots",
    "away_shots",
    "home_sot",
    "away_sot",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
]
EXTRA_COLUMNS = ODDS_COLUMNS + STAT_COLUMNS

# Mapowanie kolumn CSV -> baza (z fallbackami)
ODDS_CSV_MAP = [
    ("avg_h", ["AvgH", "B365H", "BWH", "WHH"]),
    ("avg_d", ["AvgD", "B365D", "BWD", "WHD"]),
    ("avg_a", ["AvgA", "B365A", "BWA", "WHA"]),
]
STAT_CSV_MAP = [
    ("home_shots", ["HS"]),
    ("away_shots", ["AS"]),
    ("home_sot", ["HST"]),
    ("away_sot", ["AST"]),
    ("home_corners", ["HC"]),
    ("away_corners", ["AC"]),
    ("home_fouls", ["HF"]),
    ("away_fouls", ["AF"]),
]

MATCH_DB_COLUMNS = [
    "div",
    "season",
    "date",
    "home_team",
    "away_team",
    "fthg",
    "ftag",
    "ftr",
    *EXTRA_COLUMNS,
]

# Maksymalna przerwa (dni) między meczami w tej samej kolejce
ROUND_MAX_GAP_DAYS = 4

DEFAULT_MODEL = "auto"

# Dixon-Coles
DC_TIME_DECAY_XI = 0.006  # waga historycznych meczów (nowsze = ważniejsze)
DC_MAX_GOALS = 10         # maks. bramek w siatce wyników
DC_OPTIMIZER_MAXITER = 150  # limit iteracji optymalizatora (szybszy trening DC)

# XGBoost – parametry klasyfikatora wyniku meczu (H/D/A)
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 4,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
}

# MLP (scikit-learn) — te same cechy co XGBoost, wymaga skalowania
MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "alpha": 0.0001,
    "learning_rate_init": 0.001,
    "max_iter": 400,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "random_state": 42,
}
