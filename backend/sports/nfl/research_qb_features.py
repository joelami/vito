"""
THROWAWAY research script -- tests the QB-level trailing features
(sports/nfl/qb_features.py) against NFL's current production
ML_FEATURE_COLS, via this project's standard Hypothesis framework.

MOTIVATION: direct answer to the "ground-up model design" question raised
this session (app owner, 2026-09-14) -- core/factor_taxonomy.py's
REGISTRY found NFL has ZERO PLAYER-category features. A backup QB
starting is plausibly a bigger single-game swing factor than almost
anything in the current TEAM-level feature set; qb_features.py's
qb_epa_trail (the proxy-starter's own trailing EPA/dropback, not the
whole offense blended together) and qb_continuity_flag (was the last two
completed games' starter the same person -- knowable live, unlike "who's
starting THIS game," see that module's own docstring) are the first real
test of that gap.

Run:  python -m sports.nfl.research_qb_features   (from backend/)
Requires Datasets/NFL/nflfastr_qb_starts.csv -- run
scripts/fetch_nflfastr_qb_starts.py once if it doesn't exist.
"""

import json
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, backtest, live_results
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.nfl import config as nfl_config
from sports.nfl.loader import load_games
from sports.nfl.weather import attach_weather
from sports.nfl.features import build_features, ML_FEATURE_COLS
from sports.nfl.nflfastr_features import build_trailing_epa_features
from sports.nfl.qb_features import build_qb_continuity_features


def hr(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def run_pipeline(feats, feature_cols) -> dict:
    wf = walk_forward_predict(feats, feature_cols)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])

    stds = ensemble.compute_residual_stds(oos, nfl_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()  # plain default -- NOT the adopted WEIGHT_ELO_SPREAD override, to isolate this test's own effect
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0)
    bets = backtest.run_backtest(oos, stds, nfl_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg, sport="NFL")
    if bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "bets": 0}
    summary = backtest.summarize(bets)
    return {"margin_corr": margin_corr, "total_corr": total_corr,
            "roi_pct": float(summary["roi_pct"].iloc[0]),
            "roi_stderr_pct": float(summary["roi_stderr_pct"].iloc[0]),
            "bets": int(summary["bets"].iloc[0])}


def main():
    hr("BUILDING BASE PIPELINE (current production state, EPA already included)")
    games = load_games()
    games = attach_weather(games)
    games_for_ratings = live_results.with_live_results(games, "NFL")
    rr = compute_power_ratings(
        games_for_ratings, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date", neutral_col="is_neutral_venue",
        config=PowerRatingConfig(
            k_factor=nfl_config.ELO_K_FACTOR, start_rating=nfl_config.ELO_START_RATING,
            home_field_adv=nfl_config.HOME_FIELD_ADV_ELO, season_regression=nfl_config.SEASON_REGRESSION,
            mov_mult_base=nfl_config.MOV_MULT_BASE, mov_mult_divisor=nfl_config.MOV_MULT_DIVISOR,
        ),
    )
    feats_base = build_features(games, rr.history)
    feats_base = build_trailing_epa_features(feats_base)  # current production baseline already has this adopted feature
    print(f"Base feature rows: {len(feats_base):,}, base ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}")

    hr("BASELINE RUN (current production ML_FEATURE_COLS)")
    baseline = run_pipeline(feats_base, ML_FEATURE_COLS)
    print(baseline)

    hr("HYPOTHESIS: adding real QB-level trailing features (first PLAYER-category signal)")
    feats_qb = build_qb_continuity_features(feats_base)
    qb_cols = [f"{side}_qb_{c}" for side in ("home", "away") for c in ("epa_trail", "continuity_flag")]
    coverage = (feats_qb["season"] >= 2007).mean()
    variant_cols = ML_FEATURE_COLS + qb_cols
    variant = run_pipeline(feats_qb, variant_cols)
    print(variant)

    hyp = Hypothesis(
        name="nfl_qb_trailing_epa_and_continuity",
        reasoning=(
            "core/factor_taxonomy.py's REGISTRY found NFL has ZERO PLAYER-category features -- every "
            "existing signal is team-level, which can't distinguish a team playing well BECAUSE its QB "
            "is playing well from a team playing well around a mediocre QB, or capture a real QB change "
            "at all. qb_epa_trail is the proxy-starter's own trailing EPA/dropback (not blended with the "
            "whole offense's rushing plays the way team-level off_epa_per_play is); qb_continuity_flag "
            "asks whether the team's last two completed games had the same starter -- a real, externally-"
            "motivated proxy for roster stability/QB uncertainty, phrased so it's knowable live (see "
            "qb_features.py's own docstring for why 'this game's actual starter' is deliberately never "
            "used). Tested as an addition ON TOP of the already-adopted EPA/success-rate features, not "
            "standalone, across every market."
        ),
        sport="NFL",
    )
    result = evaluate_hypothesis(hyp, baseline, variant)
    hr("HYPOTHESIS RESULT (full sample)")
    print(f"real QB-start coverage (season>=2007, has a real prior-game trail): {coverage*100:.1f}%")
    print(json.dumps(result.to_dict(), indent=2))

    hr("SPLIT-HALF STABILITY CHECK -- required before trusting the full-sample result above")
    seasons = sorted(feats_qb["season"].unique())
    mid = seasons[len(seasons) // 2]
    for label, sub in [("first half", feats_qb[feats_qb["season"] < mid]),
                        ("second half", feats_qb[feats_qb["season"] >= mid])]:
        b = run_pipeline(sub, ML_FEATURE_COLS)
        v = run_pipeline(sub, variant_cols)
        print(f"{label}: margin {b['margin_corr']:.4f}->{v['margin_corr']:.4f} "
              f"({v['margin_corr']-b['margin_corr']:+.4f}), "
              f"total {b['total_corr']:.4f}->{v['total_corr']:.4f} ({v['total_corr']-b['total_corr']:+.4f})")
    print(
        "\nREAL RESULT (checked 2026-09-15): margin_corr degrades in BOTH halves "
        "(first -0.0061, past the noise floor; second -0.0047, within it) and total_corr "
        "FLIPS SIGN between halves (+0.0043 first, -0.0030 second) -- the full-sample "
        "'adopt' above does not survive its own split-half check and is NOT a stable, "
        "real effect. RECOMMENDATION: reject, not adopt -- logged honestly in "
        "decision_log.jsonl rather than trusting the full-sample aggregate. This is kept "
        "as real, working, tested infrastructure (the fetch script, qb_features.py, and "
        "its own unit tests) since the underlying data and walk-forward-safety are sound "
        "-- only the CURRENT feature formulation (5-game trailing EPA + 2-game continuity "
        "flag) failed to show a stable improvement. A different formulation (longer/shorter "
        "trailing window, or a real depth-chart-based injury signal instead of a start-"
        "continuity proxy) is a legitimate future re-test, not a dead end."
    )
    hr("DONE")


if __name__ == "__main__":
    main()
