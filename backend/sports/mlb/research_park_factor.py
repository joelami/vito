"""
RESEARCH SCRIPT -- not wired into build_features/ML_FEATURE_COLS by default
(pending the evaluate_hypothesis() outcome logged below / in decision_log.jsonl).

Tests one hypothesis via core.research.evaluate_hypothesis: whether a real
ballpark scoring-environment factor (sports/mlb/park_factor.attach_park_
factor) carries incremental signal on top of the current production MLB
feature set. See park_factor.py's own module docstring for the full
reasoning/sourcing/relocation-safety verification -- not repeated here.

Tested ALONE (not combined with defensive_efficiency's DER hypothesis, a
SEPARATE test in research_defensive_efficiency.py) -- per this task's
explicit instruction: testing the two together would leave it unknown
which one (if either) is actually doing the work if only one helps.

Includes a REQUIRED split-half stability check, not just the full-sample
evaluate_hypothesis() result -- see sports/nfl/research_qb_features.py's
own docstring/decision_log.jsonl entry for exactly why a full-sample-only
"adopt" is not trustworthy on its own (a real full-sample "adopt" from
that session did not survive this exact check), and sports/nfl/research_
nflfastr_epa_sos_adjustment.py for the pattern this script mirrors.

Run:  python -m sports.mlb.research_park_factor   (from backend/)
"""

import json
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, backtest, live_results
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.mlb import config as mlb_config
from sports.mlb.loader import load_games
from sports.mlb.odds_loader import attach_odds
from sports.mlb.starting_pitcher import attach_starter_quality
from sports.mlb.starter_kbb_quality import attach_starter_kbb_pct
from sports.mlb.features import build_features, ML_FEATURE_COLS
from sports.mlb.park_factor import attach_park_factor


def hr(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def run_pipeline(feats, feature_cols) -> dict:
    wf = walk_forward_predict(feats, feature_cols, min_train_seasons=3)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])

    stds = ensemble.compute_residual_stds(oos, mlb_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, allowed_confidence=("Medium", "High"), price_point="Close")
    bets = backtest.run_backtest(oos, stds, mlb_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg, sport="MLB")
    if bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "bets": 0}
    summary = backtest.summarize(bets)
    return {"margin_corr": margin_corr, "total_corr": total_corr,
            "roi_pct": float(summary["roi_pct"].iloc[0]),
            "roi_stderr_pct": float(summary["roi_stderr_pct"].iloc[0]),
            "bets": int(summary["bets"].iloc[0])}


def build_all_feats():
    games = load_games()
    games = attach_odds(games)
    games = attach_starter_quality(games)       # already-adopted -- part of BOTH baseline and variant
    games = attach_starter_kbb_pct(games)        # already-adopted -- part of BOTH baseline and variant
    games_for_ratings = live_results.with_live_results(games, "MLB")
    rr = compute_power_ratings(
        games_for_ratings, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date",
        config=PowerRatingConfig(
            k_factor=mlb_config.ELO_K_FACTOR, start_rating=mlb_config.ELO_START_RATING,
            home_field_adv=mlb_config.HOME_FIELD_ADV_ELO, season_regression=mlb_config.SEASON_REGRESSION,
            mov_mult_base=mlb_config.MOV_MULT_BASE, mov_mult_divisor=mlb_config.MOV_MULT_DIVISOR,
        ),
    )
    feats = build_features(games, rr.history)
    feats = attach_park_factor(feats)  # adds park_scoring_factor -- present in feats either way, just not in ML_FEATURE_COLS unless variant tests it
    return feats


def main():
    hr("BUILDING FEATURES (current production MLB feature set + park_scoring_factor column)")
    feats = build_all_feats()
    print(f"Feature rows: {len(feats):,}, base ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}")
    print(feats["park_scoring_factor"].describe().to_string())

    hr("BASELINE RUN (current production ML_FEATURE_COLS -- no park factor)")
    baseline = run_pipeline(feats, ML_FEATURE_COLS)
    print(baseline)

    hr("HYPOTHESIS: adding park_scoring_factor")
    variant_cols = ML_FEATURE_COLS + ["park_scoring_factor"]
    variant = run_pipeline(feats, variant_cols)
    print(variant)

    hyp = Hypothesis(
        name="mlb_park_scoring_factor",
        reasoning=(
            "Park factors are one of the oldest, most universally accepted adjustments in sabermetrics "
            "(Bill James popularized the idea; every modern site -- FanGraphs, Baseball Reference, "
            "Statcast -- publishes one). Two teams can have identical trailing scoring stats while "
            "playing in parks with genuinely different real run environments (Coors Field vs. a "
            "pitcher's park), and nothing in the current MLB feature set adjusts for that at all. This "
            "project's own MLB loader already carries a park_id column (used previously for the "
            "travel_fatigue_short_rest_park_change feature) -- this needs no new data source, just a "
            "trailing park-vs-league scoring-rate computation from games already loaded, the same "
            "'compute a trailing rate, shift/recenter' pattern every other rolling feature here uses. "
            "Tested alone (not combined with the separate DER hypothesis) so it's clear which signal, "
            "if either, is actually doing the work."
        ),
        sport="MLB",
    )
    result = evaluate_hypothesis(hyp, baseline, variant)
    hr("HYPOTHESIS RESULT (full sample)")
    print(json.dumps(result.to_dict(), indent=2))

    hr("SPLIT-HALF STABILITY CHECK -- required before trusting the full-sample result above")
    seasons = sorted(feats["season"].dropna().unique())
    mid = seasons[len(seasons) // 2]
    stable = True
    for label, sub in [(f"early ({seasons[0]}-{mid - 1})", feats[feats["season"] < mid]),
                        (f"late ({mid}-{seasons[-1]})", feats[feats["season"] >= mid])]:
        b = run_pipeline(sub, ML_FEATURE_COLS)
        v = run_pipeline(sub, variant_cols)
        md = v["margin_corr"] - b["margin_corr"]
        td = v["total_corr"] - b["total_corr"]
        print(f"{label}: n={len(sub):,} margin {b['margin_corr']:.4f}->{v['margin_corr']:.4f} ({md:+.4f}), "
              f"total {b['total_corr']:.4f}->{v['total_corr']:.4f} ({td:+.4f})")
        if md < -0.005 or td < -0.005:
            stable = False
    print(f"\n{'STABLE' if stable else 'NOT STABLE'} across both halves (neither corr degraded past the noise floor "
          f"in either half) -- {'trust' if stable else 'do NOT trust'} the full-sample recommendation above.")
    hr("DONE")


if __name__ == "__main__":
    main()
