"""
Scores a single brand-new NFL matchup (not yet in the historical dataset)
against the model — shared by `main.py` (manually-entered games) and
`harness.py` (ESPN-synced real games), so both go through identical feature
construction and there's exactly one place this logic can have a bug.
"""

import pandas as pd

from . import config as nfl_config
from .features import naive_score_features, pythagorean_win_pct
from .weather import current_stadium_row


def build_matchup_feature_row(pipeline: dict, home_team: str, away_team: str, game_date,
                               is_playoff: bool = False, is_neutral_venue: bool = False) -> pd.DataFrame:
    """One-row feature DataFrame for a hypothetical/upcoming game, built from
    the pipeline's current power ratings and each team's latest rolling form —
    the same feature schema `core.ml_models.FinalModels.predict()` expects."""
    home_fr = nfl_config.canonical_franchise(home_team)
    away_fr = nfl_config.canonical_franchise(away_team)
    ratings = pipeline["current_ratings"]
    form = pipeline["current_form"]

    home_rating = ratings.get(home_fr, nfl_config.ELO_START_RATING)
    away_rating = ratings.get(away_fr, nfl_config.ELO_START_RATING)

    # ESPN-sourced dates are tz-aware (UTC, e.g. "2026-09-10T23:00Z"); manually-entered
    # and historical dates are naive. Normalize both through UTC then drop tzinfo so the
    # rest-days subtraction below never mixes aware/naive timestamps (pandas raises on that).
    game_date = pd.to_datetime(game_date, utc=True).tz_localize(None)
    home_row = form.loc[home_fr] if home_fr in form.index else None
    away_row = form.loc[away_fr] if away_fr in form.index else None

    def rest_days(team_row):
        if team_row is None or pd.isna(team_row["last_game_date"]):
            return 7.0
        return max((game_date - team_row["last_game_date"]).days, 0)

    home_pf = home_row["pf_l10"] if home_row is not None else 22.0
    home_pa = home_row["pa_l10"] if home_row is not None else 22.0
    away_pf = away_row["pf_l10"] if away_row is not None else 22.0
    away_pa = away_row["pa_l10"] if away_row is not None else 22.0
    naive_total, naive_margin = naive_score_features(home_pf, home_pa, away_pf, away_pa)

    home_rest = rest_days(home_row)
    away_rest = rest_days(away_row)
    # historical archive weather doesn't exist for a game that hasn't happened yet — neutral
    # fallback (documented limitation, see weather.current_stadium_row's docstring)
    season = nfl_config.season_for_date(game_date)
    weather = current_stadium_row(home_fr, season, game_date)
    return pd.DataFrame([{
        "rating_diff_pre": home_rating - away_rating,
        "rest_diff": home_rest - away_rest,
        "home_rest_days": home_rest, "away_rest_days": away_rest,
        "home_ats_pct_l10": home_row["ats_pct_l10"] if home_row is not None else 0.5,
        "away_ats_pct_l10": away_row["ats_pct_l10"] if away_row is not None else 0.5,
        "home_win_pct_l10": home_row["win_pct_l10"] if home_row is not None else 0.5,
        "away_win_pct_l10": away_row["win_pct_l10"] if away_row is not None else 0.5,
        "home_pf_l10": home_pf, "home_pa_l10": home_pa, "away_pf_l10": away_pf, "away_pa_l10": away_pa,
        "naive_total": naive_total, "naive_margin": naive_margin,
        "home_pyth_pct": pythagorean_win_pct(home_pf, home_pa),
        "away_pyth_pct": pythagorean_win_pct(away_pf, away_pa),
        "pyth_pct_diff": pythagorean_win_pct(home_pf, home_pa) - pythagorean_win_pct(away_pf, away_pa),
        "home_streak": home_row["streak"] if home_row is not None else 0,
        "away_streak": away_row["streak"] if away_row is not None else 0,
        "is_divisional": nfl_config.is_divisional_game(home_fr, away_fr),
        "is_playoff": is_playoff, "is_neutral_venue": is_neutral_venue,
        "game_temp_f": weather["game_temp_f"], "game_wind_mph": weather["game_wind_mph"],
        "game_precip_mm": weather["game_precip_mm"], "is_dome": weather["is_dome"],
    }]), home_rating, away_rating, naive_total


def score_matchup(pipeline: dict, home_team: str, away_team: str, game_date,
                   market_odds: dict, is_playoff: bool = False, is_neutral_venue: bool = False,
                   price_point: str = "Close") -> list:
    """
    `market_odds` must use the same Open/Close column-name convention as
    `core.edge_finder.evaluate_game` (e.g. `{"Home Odds Close": 1.91, ...}`)
    for whichever `price_point` is passed. Returns a list of
    `core.edge_finder.BetOpportunity`.
    """
    from core import edge_finder  # local import: avoids a core<->sports circular import at module load

    feat_row, home_rating, away_rating, naive_total = build_matchup_feature_row(
        pipeline, home_team, away_team, game_date, is_playoff, is_neutral_venue
    )
    pred = pipeline["final_models"].predict(feat_row).iloc[0]

    model_row = dict(market_odds)
    model_row.update({
        "rating_diff_pre": home_rating - away_rating,
        "predicted_margin": pred["predicted_margin"], "predicted_total": pred["predicted_total"],
        "naive_total": naive_total,
    })
    return edge_finder.evaluate_game(model_row, pipeline["stds"], nfl_config.ELO_POINTS_PER_MARGIN,
                                      pipeline["ensemble_cfg"], price_point=price_point)
