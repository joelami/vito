"""
Second hypothesis-test driver, same discipline as
_hypothesis_test_starting_pitcher.py. BASELINE here is the CURRENT live
feature set (ML_FEATURE_COLS as it stands today, which already includes the
adopted starting-pitcher rolling ER/start feature) - VARIANT adds bullpen
fatigue (home_bullpen_apps_l3d/away_bullpen_apps_l3d/bullpen_fatigue_diff)
on top of that. Left in place after the run as a record, same as the first
driver; not imported by features.py/config.py, no effect on the live
pipeline either way.

Run with: python -m sports.mlb._hypothesis_test_bullpen_fatigue (from backend/)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, backtest
from core.research import Hypothesis, evaluate_hypothesis
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from sports.mlb import config as mlb_config
from sports.mlb.loader import load_games
from sports.mlb.odds_loader import attach_odds
from sports.mlb.starting_pitcher import attach_starter_quality
from sports.mlb.features import build_features, ML_FEATURE_COLS
from sports.mlb.research_bullpen_fatigue import attach_bullpen_fatigue


def run_pipeline(feature_cols, add_bullpen: bool):
    games = load_games()
    games = attach_odds(games)
    games = attach_starter_quality(games)  # already-adopted feature, part of BOTH baseline and variant now

    rating_cfg = PowerRatingConfig(
        k_factor=mlb_config.ELO_K_FACTOR,
        start_rating=mlb_config.ELO_START_RATING,
        home_field_adv=mlb_config.HOME_FIELD_ADV_ELO,
        season_regression=mlb_config.SEASON_REGRESSION,
        mov_mult_base=mlb_config.MOV_MULT_BASE,
        mov_mult_divisor=mlb_config.MOV_MULT_DIVISOR,
    )
    rr = compute_power_ratings(
        games, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date",
        config=rating_cfg,
    )

    feats = build_features(games, rr.history)

    if add_bullpen:
        feats = attach_bullpen_fatigue(feats)
        feats["bullpen_fatigue_diff_l3d"] = feats["away_bullpen_apps_l3d"] - feats["home_bullpen_apps_l3d"]
        # positive = away's bullpen more heavily used recently than home's -> favors home

    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    def corr(a, b):
        return float(np.corrcoef(oos[a], oos[b])[0, 1])

    margin_corr = corr("predicted_margin", "actual_margin")
    total_corr = corr("predicted_total", "actual_total")

    stds = ensemble.compute_residual_stds(oos, mlb_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(
        min_edge_pct=3.0,
        allowed_confidence=("Medium", "High"),
        price_point="Close",
    )
    bets = backtest.run_backtest(oos, stds, mlb_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    if bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "n_bets": 0}

    summary = backtest.summarize(bets)
    row = summary.iloc[0]
    return {
        "margin_corr": margin_corr, "total_corr": total_corr,
        "roi_pct": float(row["roi_pct"]), "roi_stderr_pct": float(row["roi_stderr_pct"]),
        "n_bets": int(row["bets"]),
    }


def main():
    print("=" * 78)
    print("BASELINE: current live ML_FEATURE_COLS (already includes starter-quality feature)")
    print("=" * 78)
    baseline = run_pipeline(ML_FEATURE_COLS, add_bullpen=False)
    print(baseline)

    print("\n" + "=" * 78)
    print("VARIANT: + home_bullpen_apps_l3d, away_bullpen_apps_l3d, bullpen_fatigue_diff_l3d")
    print("=" * 78)
    variant_cols = ML_FEATURE_COLS + ["home_bullpen_apps_l3d", "away_bullpen_apps_l3d", "bullpen_fatigue_diff_l3d"]
    variant = run_pipeline(variant_cols, add_bullpen=True)
    print(variant)

    hyp = Hypothesis(
        name="bullpen_fatigue_relief_appearances_l3d",
        reasoning=(
            "Team bullpen-management practice widely caps most relievers at ~2-3 "
            "consecutive days of work specifically because performance (command, "
            "velocity) degrades with back-to-back-day usage - a 'gassed' bullpen is a "
            "real, well-documented factor in how a team's relief corps performs on a "
            "given day, and it is qualitatively different from the starting-pitcher "
            "quality feature already adopted (which only reflects the starter's own "
            "recent performance, not the relief corps backing him up). This signal is "
            "currently completely absent from the MLB feature set. The same Retrosheet "
            "event files already parsed for the starter feature also record every "
            "pitching change (`sub` lines with fieldpos==1), letting us build a "
            "walk-forward-safe team-level trailing-3-day relief-appearance count as an "
            "honest proxy for bullpen fatigue (not innings/pitches - an appearance "
            "count, deliberately scoped the same way the starter ER/start feature was)."
        ),
        sport="MLB",
    )

    result = evaluate_hypothesis(hyp, baseline, variant)
    print("\n" + "=" * 78)
    print("HYPOTHESIS RESULT")
    print("=" * 78)
    import json
    print(json.dumps(result.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
