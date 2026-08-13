"""
Sport-agnostic Elo-style power rating engine. Margin-of-victory adjusted,
home-field aware, with regression toward the mean at season boundaries.

Critical property this module exists to guarantee: for every game, the
rating used to *predict* that game (`home_rating_pre` / `away_rating_pre`)
reflects only games strictly before it in chronological order. Ratings are
updated AFTER a game is processed, never before. This is what makes the
downstream backtest walk-forward-honest rather than leaking future results
into a "prediction."

Formula is the standard MOV-adjusted Elo used publicly for NFL forecasting
(the "538 Elo" approach): expected score from a logistic curve on the rating
gap (home rating gets a home-field bonus added before comparing), rating
change scaled by a margin-of-victory multiplier so blowouts move ratings
more than narrow wins, with diminishing effect the bigger the pre-game
favorite already was.
"""

import math
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PowerRatingConfig:
    k_factor: float = 20.0
    start_rating: float = 1500.0
    home_field_adv: float = 48.0
    season_regression: float = 0.33
    mov_mult_base: float = 2.2
    mov_mult_divisor: float = 2.2


def _expected_home(home_pre: float, away_pre: float, hfa: float) -> float:
    diff = (home_pre + hfa) - away_pre
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def _mov_multiplier(margin: float, elo_diff_with_hfa: float, base: float, divisor: float) -> float:
    if margin == 0:
        return 1.0  # rare tie: still register the "surprise" factor, log-based formula breaks down at 0
    winner_diff = elo_diff_with_hfa if margin > 0 else -elo_diff_with_hfa
    mult = math.log(abs(margin) + 1.0) * (base / (winner_diff * 0.001 + divisor))
    return max(mult, 0.05)  # floor to avoid a near-zero/negative multiplier on extreme upsets


@dataclass
class RatingResult:
    history: pd.DataFrame                 # one row per game: pre-game ratings + elo win prob
    final_ratings: dict                   # team -> rating as of the last game processed
    last_season_seen: dict                # team -> season number of their most recent game
    config: PowerRatingConfig = field(default_factory=PowerRatingConfig)

    def ratings_entering_season(self, season: int) -> dict:
        """
        Snapshot of every team's rating as it would stand entering `season`,
        applying the one-time season-boundary regression to teams whose last
        game was in an earlier season. Used both to score the first games of
        a new season during backtesting and to power the "current ratings"
        UI table before a new season's games exist yet.
        """
        snap = {}
        for team, rating in self.final_ratings.items():
            last_season = self.last_season_seen.get(team)
            if last_season is not None and last_season < season:
                rating = rating + self.config.season_regression * (self.config.start_rating - rating)
            snap[team] = rating
        return snap


def compute_power_ratings(
    games: pd.DataFrame,
    home_col: str,
    away_col: str,
    home_score_col: str,
    away_score_col: str,
    season_col: str,
    date_col: str = "date",
    neutral_col: str = None,
    config: PowerRatingConfig = None,
) -> RatingResult:
    """
    `games` must already be sorted chronologically ascending (loaders are
    responsible for this). Returns pre-game ratings for every row plus final
    ratings per team.
    """
    cfg = config or PowerRatingConfig()
    games = games.sort_values(date_col, kind="stable")

    ratings = {}
    last_season = {}
    rows = []

    for _, g in games.iterrows():
        home, away = g[home_col], g[away_col]
        season = g[season_col]

        for team in (home, away):
            if team not in ratings:
                ratings[team] = cfg.start_rating
                last_season[team] = season
            elif last_season[team] < season:
                ratings[team] = ratings[team] + cfg.season_regression * (cfg.start_rating - ratings[team])
                last_season[team] = season

        # neutral-site games (Super Bowl, international games) get no home-field bonus —
        # the "home team" label there is a nominal designation, not an actual home crowd
        hfa = 0.0 if (neutral_col is not None and bool(g[neutral_col])) else cfg.home_field_adv

        home_pre, away_pre = ratings[home], ratings[away]
        expected_home = _expected_home(home_pre, away_pre, hfa)

        margin = g[home_score_col] - g[away_score_col]
        actual_home = 0.5 if margin == 0 else (1.0 if margin > 0 else 0.0)

        elo_diff_with_hfa = (home_pre + hfa) - away_pre
        mov_mult = _mov_multiplier(margin, elo_diff_with_hfa, cfg.mov_mult_base, cfg.mov_mult_divisor)

        delta = cfg.k_factor * mov_mult * (actual_home - expected_home)
        ratings[home] = home_pre + delta
        ratings[away] = away_pre - delta

        rows.append({
            "game_id": g.get("game_id"),
            "home_rating_pre": home_pre,
            "away_rating_pre": away_pre,
            "rating_diff_pre": home_pre - away_pre,
            "elo_home_win_prob": expected_home,
        })

    history = pd.DataFrame(rows).set_index("game_id")
    return RatingResult(history=history, final_ratings=ratings, last_season_seen=last_season, config=cfg)
