"""
THROWAWAY research script -- NOT part of the production pipeline, not
imported by anything else. Tests a hypothesis specifically enabled by NHL
Edge (github.com/coreyjs/nhl-api-py wrapping api-web.nhle.com's real,
undocumented tracking-data platform -- see sports/nhl/nhl_api_client.py's
module docstring for the verified coverage this test is built around).

MOTIVATION: `research_starting_goalie_save_pct.py` already tested (and
"watch"-listed, see decision_log.jsonl/subgroup_watchlist.jsonl,
2026-09-03) the starting goalie's own IN-SEASON RECENCY save% (save_pct_l10,
a rolling window of the goalie's last 10 appearances). That signal answers
"is this goalie hot or cold right now" but says nothing about shot
DIFFICULTY -- a goalie facing a heavier mix of high-danger chances will show
a lower raw save% than an equally-skilled goalie facing mostly low-danger
shots, purely from shot mix, not true ability. A generic box score (or
MoneyPuck's per-goalie shot log, which has shots/goals but no shot-LOCATION
attribution) cannot separate the two. NHL Edge's real per-goalie shot-
location breakdown (`edge.goalie_shot_location_detail`) can: it reports
shots-against/saves/save% split out by ice-area ("Crease", "High Slot",
"Low Slot", "L Circle", etc.), letting a HIGH-DANGER-SPECIFIC save% be
computed -- a genuinely different piece of information than "how many total
shots did this goalie stop lately," and the kind of shot-quality-adjusted
goaltending metric hockey analytics (e.g. Corsica/MoneyPuck's own xG models)
treats as a real refinement over plain save%.

REAL COVERAGE CONSTRAINT (see nhl_api_client.py -- confirmed live, not
assumed): Edge's per-goalie detail endpoints only exist for seasons
2021-22 through 2025-26 (5 season labels: 2022-2026 in this project's own
season-labeling convention), and only for goalies who cleared Edge's own
`minimumGamesPlayed` reporting threshold that season (25 games, confirmed).
This feature is therefore built the same walk-forward-safe way as this
project's "star player" research already established (see
research_player_matchup.py's STEP 2): a starting goalie's PRIOR SEASON's
Edge high-danger save% is used as a fixed, once-a-season-updating skill
estimate for the CURRENT season -- never the same season being scored (no
leakage), and never requiring PER-GAME Edge granularity, which the public
Edge detail endpoints do not expose (verified: `skater_skating_speed_detail`
and `goalie_shot_location_detail` both return SEASON-level aggregates only,
not a per-game log -- see nhl_api_client.py). Only seasons 2023-2026 (whose
PRIOR season, 2022-2025, is Edge-covered) can carry a real, non-fallback
value; every earlier season and every goalie below the games-played bar
gets the league-average high-danger save% -- an honest, small, real test
population, not the full history.

Run with:  python -m sports.nhl.research_edge_goalie_high_danger_save_pct   (from backend/)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/ on path
warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, backtest, research
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.nhl import config as nhl_config
from sports.nhl.loader import load_games
from sports.nhl.features import build_features, ML_FEATURE_COLS
from sports.nhl.nhl_api_client import edge_goalie_shot_location_detail, EDGE_SEASONS
from sports.nhl.research_starting_goalie_save_pct import load_goalie_starter_table

# The three innermost ice-areas Edge's own taxonomy reports -- the standard
# "high-danger" zone grouping hockey analytics uses (closest to the net,
# highest real-world shooting percentage). Verified directly against a real
# goalie's full area list (17 named zones total; these three are the
# highest-shot-volume, lowest-save% zones for every goalie inspected).
HIGH_DANGER_AREAS = {"Crease", "High Slot", "Low Slot"}


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def season_label_to_edge_str(season_label: int) -> str:
    """This project's own season label (e.g. 2025 = the season ENDING in
    2025, per config.py's convention) -> Edge's own season-id string
    (e.g. "20242025")."""
    return f"{season_label - 1}{season_label}"


def fetch_prior_season_hd_save_pct(player_id: int, season_label: int, cache: dict) -> float:
    """Prior-season (season_label - 1) Edge high-danger save% for this
    goalie, or None if Edge has no data for that player/season (pre-Edge
    era, or below the games-played bar -- see module docstring). `cache`
    memoizes by (player_id, season_label) across the many games one
    starter appears in, since this hits a real live endpoint."""
    key = (player_id, season_label)
    if key in cache:
        return cache[key]

    prior_edge_str = season_label_to_edge_str(season_label - 1)
    if prior_edge_str not in EDGE_SEASONS:
        cache[key] = None
        return None

    detail = edge_goalie_shot_location_detail(player_id, season=prior_edge_str)
    areas = detail.get("shotLocationDetails") or []
    hd_shots = sum(a.get("shotsAgainst", 0) for a in areas if a.get("area") in HIGH_DANGER_AREAS)
    hd_saves = sum(a.get("saves", 0) for a in areas if a.get("area") in HIGH_DANGER_AREAS)
    value = (hd_saves / hd_shots) if hd_shots > 0 else None
    cache[key] = value
    return value


def build_hd_save_pct_feature(feats: pd.DataFrame, games: pd.DataFrame) -> tuple:
    """Returns (feats_with_feature, coverage_fraction). Reuses the exact
    same starter-identification table research_starting_goalie_save_pct.py
    already built (icetime-max proxy over MoneyPuck's per-goalie log) --
    this test is about the EDGE SHOT-LOCATION feature, not about
    re-deriving starter identity (that upgrade is this project's separate,
    already-logged research_starting_goalie_save_pct.py rebuild)."""
    starters, _ = load_goalie_starter_table()
    starters = starters.rename(columns={"gameDate": "date_int"})

    g = games[["game_id", "date", "season", "home_team_name", "away_team_name"]].copy()
    from sports.nhl.research_starting_goalie_save_pct import NAME_TO_ABBREV
    g["home_abbrev"] = g["home_team_name"].map(NAME_TO_ABBREV)
    g["away_abbrev"] = g["away_team_name"].map(NAME_TO_ABBREV)
    local_date = pd.to_datetime(g["date"]).dt.tz_localize(None).dt.normalize()
    shift_back = pd.to_datetime(g["date"]).dt.hour < 12
    local_date = local_date.where(~shift_back, local_date - pd.Timedelta(days=1))
    g["date_int"] = local_date.dt.strftime("%Y%m%d").astype(int)

    home_st = starters.rename(columns={"playerTeam": "home_abbrev", "playerId": "home_starter_id"})[
        ["date_int", "home_abbrev", "home_starter_id"]]
    away_st = starters.rename(columns={"playerTeam": "away_abbrev", "playerId": "away_starter_id"})[
        ["date_int", "away_abbrev", "away_starter_id"]]
    g = g.merge(home_st, on=["date_int", "home_abbrev"], how="left")
    g = g.merge(away_st, on=["date_int", "away_abbrev"], how="left")

    cache = {}
    league_values = []

    def lookup(row, side):
        pid = row[f"{side}_starter_id"]
        season = row["season"]
        if pd.isna(pid):
            return np.nan
        v = fetch_prior_season_hd_save_pct(int(pid), int(season), cache)
        if v is not None:
            league_values.append(v)
        return v if v is not None else np.nan

    print("Fetching real Edge high-danger save% for every distinct (starter, season) pair "
          "in a season whose PRIOR season is Edge-covered -- one live call per distinct pair "
          "(memoized), everything else is a local lookup...")
    g["home_hd_save_pct"] = g.apply(lambda r: lookup(r, "home"), axis=1)
    g["away_hd_save_pct"] = g.apply(lambda r: lookup(r, "away"), axis=1)

    league_avg = float(np.mean(league_values)) if league_values else 0.85  # realistic high-danger save% fallback
    coverage = (g["home_hd_save_pct"].notna() | g["away_hd_save_pct"].notna()).mean()
    print(f"Real (non-fallback) Edge high-danger save% coverage: {coverage*100:.2f}% of games "
          f"(only possible for seasons {sorted(EDGE_SEASONS)}' FOLLOWING season, and only for "
          f"goalies clearing Edge's games-played bar). League-average fallback: {league_avg:.4f}")
    print(f"Distinct real (player_id, season) Edge lookups performed: "
          f"{sum(1 for v in cache.values() if v is not None)} real, {sum(1 for v in cache.values() if v is None)} missed/pre-Edge")

    g["home_hd_save_pct"] = g["home_hd_save_pct"].fillna(league_avg)
    g["away_hd_save_pct"] = g["away_hd_save_pct"].fillna(league_avg)

    out = feats.merge(g[["game_id", "home_hd_save_pct", "away_hd_save_pct"]], on="game_id", how="left")
    out["home_hd_save_pct"] = out["home_hd_save_pct"].fillna(league_avg)
    out["away_hd_save_pct"] = out["away_hd_save_pct"].fillna(league_avg)
    return out, coverage


def run_pipeline(feats: pd.DataFrame, feature_cols: list) -> dict:
    """Same TOTAL-market-only convention as research_starting_goalie_save_pct.py
    -- goaltending quality most directly acts on total goals scored."""
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])

    stds = ensemble.compute_residual_stds(oos, nhl_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close")
    bets = backtest.run_backtest(oos, stds, nhl_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    total_bets = bets[bets["market"] == "total"] if not bets.empty else bets
    if total_bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "bets": 0}
    summary = backtest.summarize(total_bets)
    return {"margin_corr": margin_corr, "total_corr": total_corr,
            "roi_pct": float(summary["roi_pct"].iloc[0]),
            "roi_stderr_pct": float(summary["roi_stderr_pct"].iloc[0]),
            "bets": int(summary["bets"].iloc[0])}


def main():
    hr("BUILDING BASE PIPELINE")
    games = load_games()
    rating_cfg = PowerRatingConfig(
        k_factor=nhl_config.ELO_K_FACTOR, start_rating=nhl_config.ELO_START_RATING,
        home_field_adv=nhl_config.HOME_FIELD_ADV_ELO, season_regression=nhl_config.SEASON_REGRESSION,
        mov_mult_base=nhl_config.MOV_MULT_BASE, mov_mult_divisor=nhl_config.MOV_MULT_DIVISOR,
    )
    rr = compute_power_ratings(
        games, home_col="home_team_id", away_col="away_team_id",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", config=rating_cfg,
    )
    feats_base = build_features(games, rr.history)

    hr("ADD BASELINE STARTER SAVE_PCT_L10 (already-watched, current best goalie-aware state)")
    from sports.nhl.research_starting_goalie_save_pct import add_starter_save_pct_feature
    feats_with_recency = add_starter_save_pct_feature(feats_base, games)
    baseline_cols = ML_FEATURE_COLS + ["home_starter_save_pct_l10", "away_starter_save_pct_l10"]

    hr("BASELINE RUN (production features + in-season recency save_pct_l10, TOTAL-market bets only)")
    baseline = run_pipeline(feats_with_recency, baseline_cols)
    print(baseline)

    hr("HYPOTHESIS: adding EDGE prior-season high-danger (shot-location-adjusted) save%")
    feats_hd, coverage = build_hd_save_pct_feature(feats_with_recency, games)
    variant_cols = baseline_cols + ["home_hd_save_pct", "away_hd_save_pct"]
    variant = run_pipeline(feats_hd, variant_cols)
    print(variant)

    hyp = Hypothesis(
        name="nhl_edge_goalie_prior_season_high_danger_save_pct",
        reasoning=(
            "The already-watchlisted starting-goalie in-season recency save% (save_pct_l10) measures "
            "'is this goalie hot/cold right now' but not shot DIFFICULTY -- a goalie facing more high-danger "
            "chances shows a lower raw save% purely from shot mix, not true skill, and no box-score or "
            "MoneyPuck shot-log source in this project's historical data carries shot-LOCATION attribution "
            "to separate the two. NHL Edge's real per-goalie shot-location breakdown "
            "(edge.goalie_shot_location_detail) does: shots/saves/save% split by ice-area, letting a genuine "
            "high-danger-specific save% (Crease+High Slot+Low Slot, hockey analytics' standard high-danger "
            "zone grouping) be computed for the first time in this project. Tested as an ADDITION to the "
            "existing recency feature (not a replacement) since the two measure genuinely different things -- "
            "recent hot/cold form vs. a shot-quality-adjusted skill estimate -- and specifically against the "
            "TOTAL market, same mechanistic reasoning as the recency feature (goaltending quality most "
            f"directly moves total goals scored). Real coverage constraint (see nhl_api_client.py, confirmed "
            "live): Edge only exists for seasons 2021-22 onward and only for goalies clearing its own "
            "games-played bar, so this used the prior-season-input convention already established by this "
            "project's star-player research (walk-forward safe, no leakage) -- coverage is real but "
            f"necessarily small ({coverage*100:.1f}% of games get a real, non-fallback value)."
        ),
        sport="NHL",
    )
    result = evaluate_hypothesis(hyp, baseline, variant)
    hr("HYPOTHESIS RESULT")
    print(json.dumps(result.to_dict(), indent=2))
    hr("DONE")


if __name__ == "__main__":
    main()
