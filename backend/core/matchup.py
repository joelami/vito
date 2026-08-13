"""
Sport-agnostic matchup scoring for a brand-new (not-yet-played) game —
generalizes the pattern `sports/nfl/matchup.py` established for NFL to any
sport whose module follows the shared convention (`current_form_snapshot`,
`ML_FEATURE_COLS`, `naive_score_features`-equivalent pace baseline).

Handles the CORE feature columns every sport module shares (rating
differential, rest days, rolling ATS/win%, rolling scoring pace, streak,
naive pace baseline). Sport-specific extras beyond that core (NFL's weather
and Pythagorean features, NBA's back-to-back flag, MLB's interleague/night
flags) are filled via each sport's optional `extra_matchup_features(...)`
hook — if a sport doesn't provide one, or a specific column isn't covered,
it defaults to a neutral value (0/False) rather than crashing. That default
is a real simplification, not a hidden one: it's fine for sports that are
currently off-season (no live game being scored yet) and documented here so
it isn't mistaken for a deliberate feature.
"""

import importlib

import pandas as pd


def _sport_module(sport: str, name: str):
    return importlib.import_module(f"sports.{sport.lower()}.{name}")


def build_matchup_feature_row(sport: str, pipeline: dict, home_team: str, away_team: str,
                               game_date, is_playoff: bool = False, is_neutral_venue: bool = False) -> tuple:
    """Returns (feature_row_df, home_rating, away_rating, naive_total). Mirrors
    `sports.nfl.matchup.build_matchup_feature_row` but dispatches to whichever
    sport's config/features module applies."""
    config = _sport_module(sport, "config")
    features = _sport_module(sport, "features")

    canonical = getattr(config, "canonical_franchise", None) or getattr(config, "canonical_team")
    home_fr = canonical(home_team)
    away_fr = canonical(away_team)
    ratings = pipeline["current_ratings"]
    form = pipeline["current_form"]

    home_rating = ratings.get(home_fr, config.ELO_START_RATING)
    away_rating = ratings.get(away_fr, config.ELO_START_RATING)

    game_date = pd.to_datetime(game_date, utc=True).tz_localize(None)
    home_row = form.loc[home_fr] if home_fr in form.index else None
    away_row = form.loc[away_fr] if away_fr in form.index else None

    def rest_days(team_row):
        if team_row is None or pd.isna(team_row.get("last_game_date")):
            return 7.0
        return max((game_date - team_row["last_game_date"]).days, 0)

    def col(row, name, default):
        if row is None:
            return default
        return row[name] if name in row and pd.notna(row[name]) else default

    home_pf = col(home_row, "pf_l10", 0.0)
    home_pa = col(home_row, "pa_l10", 0.0)
    away_pf = col(away_row, "pf_l10", 0.0)
    away_pa = col(away_row, "pa_l10", 0.0)

    naive_score_fn = getattr(features, "naive_score_features", None)
    if naive_score_fn:
        naive_total, naive_margin = naive_score_fn(home_pf, home_pa, away_pf, away_pa)
    else:
        naive_total, naive_margin = home_pf + away_pf, home_pf - away_pf

    home_rest = rest_days(home_row)
    away_rest = rest_days(away_row)

    row = {
        "rating_diff_pre": home_rating - away_rating,
        "rest_diff": home_rest - away_rest,
        "home_rest_days": home_rest, "away_rest_days": away_rest,
        "home_ats_pct_l10": col(home_row, "ats_pct_l10", 0.5),
        "away_ats_pct_l10": col(away_row, "ats_pct_l10", 0.5),
        "home_win_pct_l10": col(home_row, "win_pct_l10", 0.5),
        "away_win_pct_l10": col(away_row, "win_pct_l10", 0.5),
        "home_pf_l10": home_pf, "home_pa_l10": home_pa, "away_pf_l10": away_pf, "away_pa_l10": away_pa,
        "naive_total": naive_total, "naive_margin": naive_margin,
        "home_streak": col(home_row, "streak", 0), "away_streak": col(away_row, "streak", 0),
        "is_playoff": is_playoff, "is_neutral_venue": is_neutral_venue,
        "is_divisional": False,
    }

    extra_fn = getattr(features, "extra_matchup_features", None)
    if extra_fn:
        row.update(extra_fn(home_fr, away_fr, game_date, home_row, away_row))

    # fill anything ML_FEATURE_COLS expects that's still missing with a neutral default
    for feat_col in features.ML_FEATURE_COLS:
        if feat_col not in row:
            row[feat_col] = False if feat_col.startswith(("is_", "home_b2b", "away_b2b")) else 0.0

    return pd.DataFrame([row]), home_rating, away_rating, naive_total


def score_matchup(sport: str, pipeline: dict, home_team: str, away_team: str, game_date,
                   market_odds: dict, is_playoff: bool = False, is_neutral_venue: bool = False,
                   price_point: str = "Close") -> list:
    """Sport-agnostic version of `sports.nfl.matchup.score_matchup`."""
    from . import edge_finder

    config = _sport_module(sport, "config")
    features = _sport_module(sport, "features")

    feat_row, home_rating, away_rating, naive_total = build_matchup_feature_row(
        sport, pipeline, home_team, away_team, game_date, is_playoff, is_neutral_venue
    )
    pred = pipeline["final_models"].predict(feat_row[features.ML_FEATURE_COLS]).iloc[0]

    model_row = dict(market_odds)
    model_row.update({
        "rating_diff_pre": home_rating - away_rating,
        "predicted_margin": pred["predicted_margin"], "predicted_total": pred["predicted_total"],
        "naive_total": naive_total,
    })
    elo_points_per_margin = config.ELO_POINTS_PER_MARGIN
    return edge_finder.evaluate_game(model_row, pipeline["stds"], elo_points_per_margin,
                                      pipeline["ensemble_cfg"], price_point=price_point)
