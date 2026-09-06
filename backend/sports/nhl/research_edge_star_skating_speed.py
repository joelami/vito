"""
THROWAWAY research script -- NOT part of the production pipeline, not
imported by anything else. A second genuinely NEW hypothesis enabled by
NHL Edge (see sports/nhl/nhl_api_client.py's module docstring for the
verified coverage this test is built around).

MOTIVATION: this project's currently-adopted NHL features (shot_diff_l10,
pp_pct_l10, pk_pct_l10 -- see features.py) are all POSSESSION/OUTCOME
signals derived from box-score events (shots, goals, power plays). None of
them measure raw physical speed/explosiveness -- a team could win the shot
and special-teams battle while being a genuinely slower, less explosive
skating team, or vice versa. NHL Edge's real skater-tracking data
(`edge.skater_skating_speed_detail`) is the first source this project has
ever had that measures athletic ability DIRECTLY (recorded on-ice skating
speed) rather than inferring it from outcomes -- a distinct mechanism
(faster teams generate more zone entries/odd-man rushes and transition
chances) worth testing on its own terms, not a repackaging of an
already-tested signal.

OPERATIONALIZATION -- reuses this project's own already-established,
walk-forward-safe "star player" convention (see research_player_matchup.py's
STEP 2, `identify_stars()`: a team's star for season S = that team's own
skater with the most total ice time in season S-1, fixed before season S
starts, non-circular). For that identified star, this test pulls their OWN
season-S-1 Edge skating-speed profile -- specifically `burstsOver22`
(the count of recorded skating bursts over 22 mph that season, converted to
its league percentile) rather than a single `maxSkatingSpeed` instant, since
a repeated ability to hit elite speed is a more robust measure of a team's
transition-game speed than one outlier burst. This is a genuinely different
signal from `research_player_matchup.py`'s own star-player test (that one
used the star's own SCORING history against a specific opponent; this one
uses a real physical-tracking measurement having nothing to do with scoring
or opponent identity at all).

REAL COVERAGE CONSTRAINT (same shape as the sibling Edge test, see
research_edge_goalie_high_danger_save_pct.py and nhl_api_client.py):
Edge only exists for seasons 2021-22 onward, and skater detail endpoints
are (like goalie ones) season-level aggregates only, not per-game -- so this
uses the star's PRIOR season profile as a fixed, once-a-season input, real
only for predicted seasons 2023-2026 (whose prior season, 2022-2025, is
Edge-covered). Earlier seasons and any star whose prior-season Edge lookup
fails get the league-average burst-frequency percentile (0.5, by
construction of a percentile).

Run with:  python -m sports.nhl.research_edge_star_skating_speed   (from backend/)
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/ on path
warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, backtest
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.nhl import config as nhl_config
from sports.nhl.loader import load_games
from sports.nhl.features import build_features, ML_FEATURE_COLS
from sports.nhl.nhl_api_client import edge_skater_skating_speed_detail, EDGE_SEASONS
from sports.nhl.research_player_matchup import load_player_games, identify_stars


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def season_label_to_edge_str(season_label: int) -> str:
    return f"{season_label - 1}{season_label}"


def fetch_prior_season_burst_percentile(player_id: int, season_label: int, cache: dict):
    """This star's OWN season-(season_label-1) Edge `burstsOver22` league
    percentile, or None if unavailable (pre-Edge, or below Edge's own
    reporting bar for that player/season). Memoized -- a real live call."""
    key = (player_id, season_label)
    if key in cache:
        return cache[key]

    prior_edge_str = season_label_to_edge_str(season_label - 1)
    if prior_edge_str not in EDGE_SEASONS:
        cache[key] = None
        return None

    detail = edge_skater_skating_speed_detail(player_id, season=prior_edge_str)
    bursts = (detail.get("skatingSpeedDetails") or {}).get("burstsOver22") or {}
    pct = bursts.get("percentile")
    cache[key] = pct
    return pct


def build_star_speed_feature(feats: pd.DataFrame, games: pd.DataFrame, stars: dict) -> tuple:
    cache = {}
    real_values = []

    def lookup(team_id, season):
        pid = stars.get((team_id, season))
        if pid is None:
            return np.nan
        v = fetch_prior_season_burst_percentile(int(pid), int(season), cache)
        if v is not None:
            real_values.append(v)
        return v if v is not None else np.nan

    print("Fetching real Edge burst-speed percentile for every distinct (star, season) pair "
          "in a season whose PRIOR season is Edge-covered (memoized, one live call per pair)...")
    g = games[["game_id", "season", "home_team_id", "away_team_id"]].copy()
    g["home_star_speed_pctile"] = g.apply(lambda r: lookup(r["home_team_id"], r["season"]), axis=1)
    g["away_star_speed_pctile"] = g.apply(lambda r: lookup(r["away_team_id"], r["season"]), axis=1)

    coverage = (g["home_star_speed_pctile"].notna() | g["away_star_speed_pctile"].notna()).mean()
    print(f"Real (non-fallback) Edge star-speed coverage: {coverage*100:.2f}% of games. "
          f"Distinct real (player,season) lookups: {sum(1 for v in cache.values() if v is not None)} real, "
          f"{sum(1 for v in cache.values() if v is None)} missed/pre-Edge.")

    g["home_star_speed_pctile"] = g["home_star_speed_pctile"].fillna(0.5)  # neutral (mid-percentile), a real percentile's own natural fallback
    g["away_star_speed_pctile"] = g["away_star_speed_pctile"].fillna(0.5)

    out = feats.merge(g[["game_id", "home_star_speed_pctile", "away_star_speed_pctile"]], on="game_id", how="left")
    out["home_star_speed_pctile"] = out["home_star_speed_pctile"].fillna(0.5)
    out["away_star_speed_pctile"] = out["away_star_speed_pctile"].fillna(0.5)
    return out, coverage


def run_pipeline(feats: pd.DataFrame, feature_cols: list) -> dict:
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")
    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])
    stds = ensemble.compute_residual_stds(oos, nhl_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close")
    bets = backtest.run_backtest(oos, stds, nhl_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    if bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "bets": 0}
    summary = backtest.summarize(bets)
    return {"margin_corr": margin_corr, "total_corr": total_corr,
            "roi_pct": float(summary["roi_pct"].iloc[0]),
            "roi_stderr_pct": float(summary["roi_stderr_pct"].iloc[0]),
            "bets": int(summary["bets"].iloc[0])}


def main():
    hr("STEP 1: LOAD + STAR IDENTIFICATION (reusing research_player_matchup.py's convention)")
    games = load_games()
    mp_games = load_player_games()
    stars = identify_stars(mp_games)

    hr("BUILDING BASE PIPELINE")
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
    print(f"Base feature rows: {len(feats_base):,}, base ML_FEATURE_COLS: {len(ML_FEATURE_COLS)} "
          f"(current production state, already includes adopted shot_diff_l10/pp_pct_l10/pk_pct_l10)")

    hr("BASELINE RUN (current production ML_FEATURE_COLS)")
    baseline = run_pipeline(feats_base, ML_FEATURE_COLS)
    print(baseline)

    hr("HYPOTHESIS: adding EDGE star skater's prior-season elite-speed-burst percentile")
    feats_speed, coverage = build_star_speed_feature(feats_base, games, stars)
    variant_cols = ML_FEATURE_COLS + ["home_star_speed_pctile", "away_star_speed_pctile"]
    variant = run_pipeline(feats_speed, variant_cols)
    print(variant)

    hyp = Hypothesis(
        name="nhl_edge_star_skater_prior_season_speed_burst_pctile",
        reasoning=(
            "Every currently-adopted NHL feature (shot_diff_l10, pp_pct_l10, pk_pct_l10) is a possession/"
            "outcome signal derived from box-score events -- none measure raw physical speed/explosiveness "
            "directly. NHL Edge's real skater-tracking data (edge.skater_skating_speed_detail) is the first "
            "source this project has ever had for that: a team's own identified star skater (this project's "
            "already-established, walk-forward-safe prior-season-TOI-leader definition, see "
            "research_player_matchup.py) generating more recorded elite-speed bursts (>22mph, a real Edge-"
            "reported count converted to league percentile) plausibly indicates more transition-game speed -- "
            "more zone entries, odd-man rushes, and end-to-end chances -- a mechanism genuinely distinct from "
            "shot volume or special-teams conversion already in the model. Used the star's own prior-season "
            "burst-frequency percentile (repeated elite-speed instances, not one outlier top-speed instant) as "
            "a fixed, once-a-season input, following the exact same walk-forward-safe convention this "
            "project's star-player research already established. Real coverage constraint (see "
            "nhl_api_client.py, confirmed live): Edge only exists from 2021-22 onward, so this is real but "
            f"necessarily small coverage ({coverage*100:.1f}% of games get a real, non-fallback value) -- an "
            "honest, disclosed limitation, not a full-history feature."
        ),
        sport="NHL",
    )
    result = evaluate_hypothesis(hyp, baseline, variant)
    hr("HYPOTHESIS RESULT")
    print(json.dumps(result.to_dict(), indent=2))
    hr("DONE")


if __name__ == "__main__":
    main()
