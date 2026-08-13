"""
One-off driver (not part of the permanent module surface) that runs the
verify.py pipeline shape twice - once with current features (baseline),
once with the starting-pitcher rolling ER/start feature added (variant) -
and calls core.research.evaluate_hypothesis. Left in place after the run as
a record of exactly how the test was executed, same spirit as keeping
research_starting_pitcher.py itself; NOT imported by features.py/config.py
and has no effect on the live pipeline either way.

Run with: python -m sports.mlb._hypothesis_test_starting_pitcher (from backend/)
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
from sports.mlb.features import build_features, ML_FEATURE_COLS
from sports.mlb.research_starting_pitcher import attach_starter_quality


def run_pipeline(feature_cols):
    games = load_games()
    games = attach_odds(games)

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

    if "home_sp_er_lN" in feature_cols:
        feats, league_avg = attach_starter_quality(feats)
        feats["home_sp_er_lN"] = feats["home_sp_er_lN"].fillna(league_avg)
        feats["away_sp_er_lN"] = feats["away_sp_er_lN"].fillna(league_avg)
        feats["sp_er_diff_lN"] = feats["away_sp_er_lN"] - feats["home_sp_er_lN"]  # positive = home's starter has been stingier (fewer ER/start) than away's -> favors home

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
    print("BASELINE: current ML_FEATURE_COLS (no starting-pitcher feature)")
    print("=" * 78)
    baseline = run_pipeline(ML_FEATURE_COLS)
    print(baseline)

    print("\n" + "=" * 78)
    print("VARIANT: + home_sp_er_lN, away_sp_er_lN, sp_er_diff_lN")
    print("=" * 78)
    variant_cols = ML_FEATURE_COLS + ["home_sp_er_lN", "away_sp_er_lN", "sp_er_diff_lN"]
    variant = run_pipeline(variant_cols)
    print(variant)

    hyp = Hypothesis(
        name="starting_pitcher_rolling_er_per_start",
        reasoning=(
            "Sabermetric consensus widely publishes starting pitcher quality as one of "
            "the single largest drivers of a single game's run environment in MLB, "
            "disproportionately more than the analogous 'who's playing' factor in NFL/CFB "
            "(a QB injury matters, but no single defensive player controls run prevention "
            "the way a starter controls innings 1-6). This signal is currently completely "
            "absent from the MLB feature set - not attempted at all. Retrosheet event files "
            "(2010seve/2020seve) newly available in this dataset let us build a walk-forward "
            "-safe rolling earned-runs-allowed-per-start proxy (not full ERA - deliberately "
            "scoped down from innings-normalized ERA since tracking exact innings pitched "
            "requires parsing substitution events, out of scope for a first pass) for both "
            "starters in a game, which nothing in the current model captures."
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
