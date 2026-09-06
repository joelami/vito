"""
THROWAWAY research script -- NOT part of the production pipeline, not
imported by anything else. Tests a real, data-motivated fix for the CFB
moneyline problem the app owner flagged (2026-09-05): CFB moneyline loses
money in every confidence tier (High hit=26.7%/ROI=-14.03%, Medium
hit=27.1%/ROI=-17.10%), and unlike NBA/NFL/MLB moneyline that isn't a
proven-backwards-ordering problem -- it's monotonic, just bad everywhere.
Bucketing that backtest by underdog price found the damage concentrated
hardest at the extreme end (market_odds 10+: 0/44, -100% ROI) but present
broadly (10-20% edge: -11.7% ROI n=162; 20-30% edge: -33.1% ROI n=119) --
see decision_log.jsonl for the full investigation.

ROOT CAUSE HYPOTHESIS: core/power_ratings.py's Elo engine gives every
never-before-seen team the exact same start_rating (1500), which is correct
for a single-tier pro league (every NFL/MLB/NBA/NHL team plays a real,
comparable schedule) but wrong for CFB, which mixes ~230 real FBS programs
with FCS/small-conference teams that only appear a handful of times as
"buy game" opponents -- the exact Abilene Christian @ Texas Tech kind of
game that started this whole investigation.

CHECKED DIRECTLY, not assumed: teams with <=8 total appearances in this
project's CFB dataset (460 distinct teams total, median 9 appearances --
this threshold sits just below that median, above the dense <=5 cluster of
clear FCS/D-II programs like Lamar, Mercer, Grambling State, Robert Morris)
lose to higher-appearance opponents by a real, sizable margin: -16.7 points
average (n=432 real rare-vs-common games), 27.5% win rate. They're the
visiting team in 80% of these games (345/432), consistent with real "money
game" structure (a small school gets paid to travel and lose). Net of the
home-field effect this project's own model already prices separately
(~6.4 points at CFB's own HOME_FIELD_ADV_ELO/ELO_POINTS_PER_MARGIN), that's
a real, sizable team-strength gap -- not just a home-field disadvantage --
that starting every team at a flat 1500 completely ignores.

THE FIX BEING TESTED: core/power_ratings.py's PowerRatingConfig gained two
new, additive, OFF-BY-DEFAULT fields this session -- low_history_start_
rating and low_history_game_threshold -- that seed (and season-regress) a
team toward a lower floor for as long as its own walk-forward-safe
games-played count (computed live, incrementally, never from full-dataset
hindsight) stays under the threshold. This script is the validation before
that gets wired into sports/cfb/config.py for real. LOW_HISTORY_START_RATING
below is derived from the real -16.7pt average margin (minus the ~6.4pt
home-field piece already handled elsewhere) via this project's own
ELO_POINTS_PER_MARGIN=14 conversion, rounded to a clean, documented value
-- not fit to make the backtest look good, and the backtest below is the
actual arbiter of whether it helps.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/ on path
warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, backtest, live_results
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.cfb import config as cfb_config
from sports.cfb.loader import load_games
from sports.cfb.features import build_features, ML_FEATURE_COLS

# (1500 - 16.7) real average margin, minus ~6.4 real points of home-field
# effect already double-counted in that raw average (rare teams are away
# 80% of the time) -> ~-12.9pt real team-strength gap -> -12.9*14 ~= -180
# Elo points. Rounded to a clean -200 (1300) -- slightly more conservative
# (lower) than the point estimate, consistent with this project's practice
# of not fitting a constant to the decimal.
LOW_HISTORY_START_RATING = 1300.0
LOW_HISTORY_GAME_THRESHOLD = 8


def hr(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def build_oos(rating_cfg: PowerRatingConfig):
    games = load_games()
    games_for_ratings = live_results.with_live_results(games, "CFB")  # real no-op for CFB, matches pipeline.py
    rr = compute_power_ratings(
        games_for_ratings, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date",
        neutral_col="is_neutral_venue" if "is_neutral_venue" in games.columns else None,
        config=rating_cfg,
    )
    feats = build_features(games, rr.history)
    wf = walk_forward_predict(feats, ML_FEATURE_COLS)
    history_df = feats.set_index("game_id").join(wf.predictions, how="left")
    return history_df.dropna(subset=["predicted_margin"])


def run_pipeline(oos_df: pd.DataFrame, market: str = None) -> dict:
    margin_corr = float(np.corrcoef(oos_df["predicted_margin"], oos_df["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos_df["predicted_total"], oos_df["actual_total"])[0, 1])

    stds = ensemble.compute_residual_stds(oos_df, cfb_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, price_point="Close")  # CFB has only one real odds snapshot
    bets = backtest.run_backtest(oos_df, stds, cfb_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg, sport="CFB")
    scoped = bets[bets["market"] == market] if (market and not bets.empty) else bets
    if scoped.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "bets": 0}
    summary = backtest.summarize(scoped)
    return {"margin_corr": margin_corr, "total_corr": total_corr,
            "roi_pct": float(summary["roi_pct"].iloc[0]),
            "roi_stderr_pct": float(summary["roi_stderr_pct"].iloc[0]),
            "bets": int(summary["bets"].iloc[0]), "hit_rate": float(summary["hit_rate"].iloc[0])}


def underdog_bucket_check(oos_df: pd.DataFrame, label: str):
    """The actual proof of concept: does the 10+ underdog-odds bucket stop
    being a 0/44 -100% ROI money pit specifically."""
    stds = ensemble.compute_residual_stds(oos_df, cfb_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, price_point="Close")
    bets = backtest.run_backtest(oos_df, stds, cfb_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg, sport="CFB")
    ml = bets[bets["market"] == "moneyline"].copy()
    ml["odds_bucket"] = pd.cut(ml["market_odds"], bins=[1, 1.5, 2, 3, 5, 10, 100],
                                labels=["<1.5", "1.5-2", "2-3", "3-5", "5-10", "10+"])
    print(f"--- {label}: CFB moneyline by underdog-odds bucket ---")
    print(backtest.summarize(ml, group_cols=["odds_bucket"]).to_string())


def main():
    hr("BASELINE: current production CFB rating config (flat 1500 start for every team)")
    baseline_cfg = PowerRatingConfig(
        k_factor=cfb_config.ELO_K_FACTOR, start_rating=cfb_config.ELO_START_RATING,
        home_field_adv=cfb_config.HOME_FIELD_ADV_ELO, season_regression=cfb_config.SEASON_REGRESSION,
        mov_mult_base=cfb_config.MOV_MULT_BASE, mov_mult_divisor=cfb_config.MOV_MULT_DIVISOR,
    )
    oos_base = build_oos(baseline_cfg)
    underdog_bucket_check(oos_base, "BASELINE")

    hr("VARIANT: low-history rating floor for teams with <=8 games played (walk-forward safe)")
    variant_cfg = PowerRatingConfig(
        k_factor=cfb_config.ELO_K_FACTOR, start_rating=cfb_config.ELO_START_RATING,
        home_field_adv=cfb_config.HOME_FIELD_ADV_ELO, season_regression=cfb_config.SEASON_REGRESSION,
        mov_mult_base=cfb_config.MOV_MULT_BASE, mov_mult_divisor=cfb_config.MOV_MULT_DIVISOR,
        low_history_start_rating=LOW_HISTORY_START_RATING,
        low_history_game_threshold=LOW_HISTORY_GAME_THRESHOLD,
    )
    oos_variant = build_oos(variant_cfg)
    underdog_bucket_check(oos_variant, "VARIANT")

    for market in ["moneyline", "spread", "total"]:
        hr(f"HYPOTHESIS ({market})")
        baseline = run_pipeline(oos_base, market)
        variant = run_pipeline(oos_variant, market)
        print("baseline:", baseline)
        print("variant :", variant)
        hyp = Hypothesis(
            name=f"cfb_low_history_rating_floor_{market}",
            reasoning=(
                f"Real data check: CFB teams with <={LOW_HISTORY_GAME_THRESHOLD} total dataset "
                f"appearances lose to common opponents by -16.7pts avg (n=432, 27.5% win rate) -- "
                f"a genuine team-strength gap the flat 1500 start_rating ignores. Testing a "
                f"walk-forward-safe (games-played computed live, never from full-history hindsight) "
                f"low-history start/regression floor of {LOW_HISTORY_START_RATING} against the "
                f"{market} market."
            ),
            sport="CFB",
        )
        result = evaluate_hypothesis(hyp, baseline, variant)
        print(json.dumps(result.to_dict(), indent=2))

    hr("DONE")


if __name__ == "__main__":
    main()
