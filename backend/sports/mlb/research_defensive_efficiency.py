"""
RESEARCH SCRIPT -- not wired into build_features/ML_FEATURE_COLS by default
(pending the evaluate_hypothesis() outcome logged below / in decision_log.jsonl).

Tests one hypothesis via core.research.evaluate_hypothesis: whether team
Defensive Efficiency Ratio (sports/mlb/defensive_efficiency.attach_
defensive_efficiency) carries incremental signal on top of the current
production MLB feature set. See defensive_efficiency.py's own module
docstring for the full reasoning/sourcing/base-code-classification
verification -- not repeated here.

Tested ALONE (not combined with park_factor's park-scoring-environment
hypothesis, a SEPARATE test in research_park_factor.py) -- per this task's
explicit instruction: testing the two together would leave it unknown
which one (if either) is actually doing the work if only one helps.

Includes a REQUIRED split-half stability check, not just the full-sample
evaluate_hypothesis() result -- see sports/nfl/research_qb_features.py's
own docstring/decision_log.jsonl entry for exactly why a full-sample-only
"adopt" is not trustworthy on its own, and sports/nfl/research_nflfastr_
epa_sos_adjustment.py for the pattern this script mirrors.

Run:  python -m sports.mlb.research_defensive_efficiency   (from backend/)
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
from sports.mlb.defensive_efficiency import attach_defensive_efficiency, load_team_game_der


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
    feats = attach_defensive_efficiency(feats)  # adds home_der_lN/away_der_lN/der_diff_lN
    return feats


def main():
    hr("PARSE EVENT FILES -> TEAM-GAME DER LOG (diagnostics)")
    der_games = load_team_game_der()
    total_bip = der_games["home_bip"].sum() + der_games["away_bip"].sum()
    total_hits = der_games["home_hits"].sum() + der_games["away_hits"].sum()
    print(f"{len(der_games):,} team-games parsed from event files. "
          f"Overall (all-time, non-walk-forward) league DER: {1 - total_hits / total_bip:.4f}")

    hr("BUILDING FEATURES (current production MLB feature set + DER columns)")
    feats = build_all_feats()
    print(f"Feature rows: {len(feats):,}, base ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}")
    print(feats[["home_der_lN", "away_der_lN", "der_diff_lN"]].describe().to_string())

    hr("BASELINE RUN (current production ML_FEATURE_COLS -- no DER)")
    baseline = run_pipeline(feats, ML_FEATURE_COLS)
    print(baseline)

    hr("HYPOTHESIS: adding home_der_lN, away_der_lN, der_diff_lN")
    variant_cols = ML_FEATURE_COLS + ["home_der_lN", "away_der_lN", "der_diff_lN"]
    variant = run_pipeline(feats, variant_cols)
    print(variant)

    hyp = Hypothesis(
        name="mlb_defensive_efficiency_ratio",
        reasoning=(
            "The already-adopted starter_kbb_pct_rolling feature's own module docstring explicitly "
            "names the gap this targets: K-BB% 'can't separate a pitcher's skill from his defense' -- "
            "it strips out defense/ballpark/sequencing-luck noise from the PITCHER's side, but nothing "
            "in ML_FEATURE_COLS measures what the DEFENSE itself contributes once a ball is actually put "
            "in play. Defensive Efficiency Ratio (DER = 1 - hits-on-balls-in-play / balls-in-play) is a "
            "real, decades-old sabermetric team-defense stat (Bill James) that is the direct, "
            "complementary answer to that named gap. The same Retrosheet event files already parsed for "
            "starting_pitcher.py, starter_kbb_quality.py, and research_bullpen_arm_quality.py carry the "
            "batted-ball/out-type information needed -- same data source, different aggregation "
            "(team-level fielding outcome, not pitcher-attributed), not a new acquisition. Tested alone "
            "(not combined with the separate park-factor hypothesis) so it's clear which signal, if "
            "either, is actually doing the work."
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
