"""
Model Dixona-Colesa dla danych z football-data.co.uk.

Wykorzystuje kolumny: Div, Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

from config import DC_MAX_GOALS, DC_MODEL_PATH, DC_TIME_DECAY_XI, DC_OPTIMIZER_MAXITER


RESULT_LABELS = ["H", "D", "A"]
RESULT_NAMES = {"H": "Wygrana gospodarzy", "D": "Remis", "A": "Wygrana gości"}


@dataclass
class LeagueParams:
    teams: list[str]
    attack: dict[str, float]
    defense: dict[str, float]
    home_adv: float
    rho: float


@dataclass
class DixonColesModel:
    leagues: dict[str, LeagueParams] = field(default_factory=dict)
    xi: float = DC_TIME_DECAY_XI
    max_goals: int = DC_MAX_GOALS

    def _team_params(
        self, params: LeagueParams, team: str
    ) -> tuple[float, float]:
        if team in params.attack:
            return params.attack[team], params.defense[team]
        avg_attack = float(np.mean(list(params.attack.values())))
        avg_defense = float(np.mean(list(params.defense.values())))
        return avg_attack, avg_defense

    def predict_proba(
        self, div: str, home_team: str, away_team: str
    ) -> dict[str, float] | None:
        params = self.leagues.get(div)
        if params is None:
            return None

        home_attack, home_defense = self._team_params(params, home_team)
        away_attack, away_defense = self._team_params(params, away_team)

        lh = np.exp(
            params.home_adv
            + home_attack
            - away_defense
        )
        la = np.exp(
            away_attack - home_defense
        )
        lh = float(np.clip(lh, 0.05, 8.0))
        la = float(np.clip(la, 0.05, 8.0))

        goals = np.arange(self.max_goals + 1)
        pmf_h = poisson.pmf(goals, lh)
        pmf_a = poisson.pmf(goals, la)
        grid = np.outer(pmf_h, pmf_a)

        rho = params.rho
        grid[0, 0] *= max(1.0 - lh * la * rho, 1e-9)
        grid[0, 1] *= max(1.0 + lh * rho, 1e-9)
        grid[1, 0] *= max(1.0 + la * rho, 1e-9)
        grid[1, 1] *= max(1.0 - rho, 1e-9)

        total = grid.sum()
        if total <= 0:
            return None
        grid /= total

        return {
            "H": float(np.tril(grid, k=-1).sum()),
            "D": float(np.trace(grid)),
            "A": float(np.triu(grid, k=1).sum()),
        }


def _vectorized_tau(
    x: np.ndarray,
    y: np.ndarray,
    lh: np.ndarray,
    la: np.ndarray,
    rho: float,
) -> np.ndarray:
    tau = np.ones_like(lh, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1.0 - lh[m00] * la[m00] * rho
    tau[m01] = 1.0 + lh[m01] * rho
    tau[m10] = 1.0 + la[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


def _fit_league(
    matches: pd.DataFrame,
    xi: float,
    max_goals: int,
) -> LeagueParams | None:
    if len(matches) < 30:
        return None

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    team_idx = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    if n < 4:
        return None

    max_date = matches["date"].max()
    hi_arr = np.array([team_idx[r.home_team] for r in matches.itertuples()], dtype=int)
    ai_arr = np.array([team_idx[r.away_team] for r in matches.itertuples()], dtype=int)
    x_arr = matches["fthg"].astype(int).to_numpy()
    y_arr = matches["ftag"].astype(int).to_numpy()
    days = np.array(
        [max((max_date - r.date).days, 0) for r in matches.itertuples()], dtype=float
    )
    w_arr = np.exp(-xi * days)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        attack = np.zeros(n)
        defense = np.zeros(n)
        attack[1:] = params[: n - 1]
        defense[1:] = params[n - 1 : 2 * (n - 1)]
        home_adv = params[2 * (n - 1)]
        rho = params[2 * (n - 1) + 1]
        return attack, defense, home_adv, rho

    def neg_log_likelihood(params: np.ndarray) -> float:
        attack, defense, home_adv, rho = unpack(params)
        lh = np.exp(home_adv + attack[hi_arr] - defense[ai_arr])
        la = np.exp(attack[ai_arr] - defense[hi_arr])
        if np.any(lh <= 0) or np.any(la <= 0):
            return 1e12

        tau = _vectorized_tau(x_arr, y_arr, lh, la, rho)
        if np.any(tau <= 0):
            return 1e12

        log_p_x = x_arr * np.log(lh) - lh - gammaln(x_arr + 1)
        log_p_y = y_arr * np.log(la) - la - gammaln(y_arr + 1)
        ll = np.sum(w_arr * (log_p_x + log_p_y + np.log(tau)))
        return float(-ll)

    x0 = np.zeros(2 * (n - 1) + 2)
    x0[-2] = 0.25
    x0[-1] = -0.03

    bounds = [(None, None)] * (2 * (n - 1)) + [(None, None), (-0.25, 0.25)]
    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": DC_OPTIMIZER_MAXITER, "ftol": 1e-4},
    )
    if not result.success:
        return None

    attack, defense, home_adv, rho = unpack(result.x)
    return LeagueParams(
        teams=teams,
        attack={teams[i]: float(attack[i]) for i in range(n)},
        defense={teams[i]: float(defense[i]) for i in range(n)},
        home_adv=float(home_adv),
        rho=float(rho),
    )


def train_dixon_coles(
    matches: pd.DataFrame,
    xi: float = DC_TIME_DECAY_XI,
    max_goals: int = DC_MAX_GOALS,
    model_path=DC_MODEL_PATH,
    verbose: bool = True,
) -> tuple[DixonColesModel, dict]:
    """Trenuje osobny model DC dla każdej ligi (Div)."""
    t0 = time.perf_counter()
    played = matches[matches["ftr"].isin(["H", "D", "A"])].copy()
    if played.empty:
        raise ValueError("Brak rozegranych meczów do treningu DC.")

    model = DixonColesModel(xi=xi, max_goals=max_goals)
    league_stats: dict[str, dict] = {}

    for div, league_matches in played.groupby("div"):
        if verbose:
            print(f"  DC: dopasowanie {div} ({len(league_matches)} meczow)...", end=" ")
        t_league = time.perf_counter()
        params = _fit_league(league_matches, xi=xi, max_goals=max_goals)
        if params is None:
            if verbose:
                print("pominieto")
            continue
        model.leagues[div] = params
        league_stats[div] = {
            "teams": len(params.teams),
            "matches": len(league_matches),
            "home_adv": round(params.home_adv, 3),
            "rho": round(params.rho, 3),
        }
        if verbose:
            print(f"OK ({time.perf_counter() - t_league:.1f}s)")

    if not model.leagues:
        raise RuntimeError("Nie udało się dopasować modelu Dixon-Coles.")

    metrics = {
        "leagues_fitted": len(model.leagues),
        "league_stats": league_stats,
        "total_matches": len(played),
        "train_seconds": round(time.perf_counter() - t0, 1),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    if verbose:
        print(f"  DC: razem {metrics['train_seconds']}s")
    return model, metrics


def load_dixon_coles(model_path=DC_MODEL_PATH) -> DixonColesModel:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model Dixon-Coles nie istnieje ({model_path}). Uruchom trening."
        )
    return joblib.load(model_path)


def predict_matches_dc(
    model: DixonColesModel,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """Prognozy H/D/A na podstawie modelu Dixon-Coles."""
    if fixtures.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for row in fixtures.itertuples():
        probs = model.predict_proba(row.div, row.home_team, row.away_team)
        if probs is None:
            continue

        predicted = max(probs, key=probs.get)
        rows.append(
            {
                "div": row.div,
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "predicted": predicted,
                "prob_H": probs["H"],
                "prob_D": probs["D"],
                "prob_A": probs["A"],
                "confidence": max(probs.values()),
                "prediction_label": RESULT_NAMES[predicted],
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["div", "date"]).reset_index(drop=True)


def evaluate_dixon_coles(
    model: DixonColesModel,
    matches: pd.DataFrame,
) -> dict:
    """Ocena dokładności na zbiorze meczów z znanym wynikiem."""
    from sklearn.metrics import accuracy_score, log_loss

    played = matches[matches["ftr"].isin(["H", "D", "A"])].copy()
    y_true: list[str] = []
    y_pred: list[str] = []
    proba_rows: list[list[float]] = []

    for row in played.itertuples():
        probs = model.predict_proba(row.div, row.home_team, row.away_team)
        if probs is None:
            continue
        y_true.append(row.ftr)
        y_pred.append(max(probs, key=probs.get))
        proba_rows.append([probs["A"], probs["D"], probs["H"]])

    if not y_true:
        return {"accuracy": 0.0, "log_loss": None, "samples": 0}

    ll = float(log_loss(y_true, proba_rows, labels=["A", "D", "H"]))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": ll,
        "samples": len(y_true),
    }
