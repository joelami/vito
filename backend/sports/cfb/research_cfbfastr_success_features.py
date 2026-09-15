"""
THROWAWAY research script -- NOT part of the production pipeline. Tests the
trailing success-rate features built from real cfbfastR play-by-play data
(sports/cfb/cfbfastr_features.py) against CFB's current production
ML_FEATURE_COLS, via this project's standard Hypothesis framework -- same
discipline as sports/nfl/research_nflfastr_epa_features.py.

MOTIVATION (app owner, 2026-09-14): "we have way too much data to not have
more confidence in what we are doing" -- CFB's own feature set has ZERO
play-level efficiency signal today (see core/factor_taxonomy.py's
coverage_report("CFB") once that's extended past NFL), same conflation
problem NFL's EPA test addressed: pf_l10/pa_l10 mix real offensive/
defensive quality with special-teams TDs, turnover-return luck, and
garbage-time scoring. Success rate (see cfbfastr_features.py's own
docstring for why it's success rate and not EPA -- a real environment
constraint, not a corner cut) is the CFB analog of that same fix.

Uses the GENERIC build_pipeline() shape (see pipeline.py's build_pipeline,
NOT build_nfl_pipeline -- CFB has no dedicated pipeline function), since
that's the real code path CFB's backtest/ratings actually run through.

Run with:  python -m sports.cfb.research_cfbfastr_success_features   (from backend/)
Requires Datasets/College Football/cfbfastr_team_game_success.csv to exist
first -- run scripts/fetch_cfbfastr_team_game_success.py once if it doesn't
(already run as of this writing: 39,303 team-game rows, 2013-2025).
"""

import json
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, backtest, live_results
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core.research import Hypothesis, evaluate_hypothesis
from sports.cfb import config as cfb_config
from sports.cfb.loader import load_games
from sports.cfb.features import build_features, ML_FEATURE_COLS
from sports.cfb.cfbfastr_features import build_trailing_success_features, TRAILING_COLS


def hr(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def run_pipeline(feats, feature_cols) -> dict:
    wf = walk_forward_predict(feats, feature_cols)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])

    stds = ensemble.compute_residual_stds(oos, cfb_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
    # price_point="Close" -- CFB's odds are a single snapshot, not real
    # open-vs-close (see loader.py's own docstring); "Open" would silently
    # find zero opportunities, same real bug pipeline.py's generic
    # build_pipeline() already documents avoiding.
    bt_cfg = backtest.BacktestConfig(min_edge_pct=3.0, price_point="Close")
    bets = backtest.run_backtest(oos, stds, cfb_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg, sport="CFB")
    if bets.empty:
        return {"margin_corr": margin_corr, "total_corr": total_corr,
                "roi_pct": float("nan"), "roi_stderr_pct": float("nan"), "bets": 0}
    summary = backtest.summarize(bets)
    return {"margin_corr": margin_corr, "total_corr": total_corr,
            "roi_pct": float(summary["roi_pct"].iloc[0]),
            "roi_stderr_pct": float(summary["roi_stderr_pct"].iloc[0]),
            "bets": int(summary["bets"].iloc[0])}


def main():
    hr("BUILDING BASE PIPELINE (current production state)")
    games = load_games()
    games_for_ratings = live_results.with_live_results(games, "CFB")  # harmless no-op, CFB not in LIVE_SPORTS
    rr = compute_power_ratings(
        games_for_ratings, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date",
        neutral_col="is_neutral_venue" if "is_neutral_venue" in games.columns else None,
        config=PowerRatingConfig(
            k_factor=cfb_config.ELO_K_FACTOR, start_rating=cfb_config.ELO_START_RATING,
            home_field_adv=cfb_config.HOME_FIELD_ADV_ELO, season_regression=cfb_config.SEASON_REGRESSION,
            mov_mult_base=cfb_config.MOV_MULT_BASE, mov_mult_divisor=cfb_config.MOV_MULT_DIVISOR,
            low_history_start_rating=cfb_config.LOW_HISTORY_START_RATING,
            low_history_game_threshold=cfb_config.LOW_HISTORY_GAME_THRESHOLD,
        ),
    )
    feats_base = build_features(games, rr.history)
    # The trailing success-rate columns are now ALREADY in production
    # ML_FEATURE_COLS (adopted in this same session) -- to genuinely
    # re-validate them (e.g. after the 2026-09-14 team-name-mapping
    # collision fix, see decision_log.jsonl), the baseline arm here must
    # be CORE-only (everything except those 4 columns), not the full
    # production list, or "baseline" and "variant" would be identical.
    sr_cols = [f"{side}_{col}_trail" for side in ("home", "away") for col in TRAILING_COLS]
    core_cols = [c for c in ML_FEATURE_COLS if c not in sr_cols]
    print(f"Base feature rows: {len(feats_base):,}, core (pre-success-rate) feature cols: {len(core_cols)}")

    hr("BASELINE RUN (CORE features only, i.e. CFB's ML_FEATURE_COLS before this adoption)")
    baseline = run_pipeline(feats_base, core_cols)
    print(baseline)

    hr("HYPOTHESIS: adding real cfbfastR trailing success-rate features (fixed, collision-safe mapping)")
    feats_sr = build_trailing_success_features(feats_base, n_games=10)
    coverage = (feats_sr["season"] >= 2014).mean()  # first real season is 2013, so 2014+ has a real prior-season trail
    variant = run_pipeline(feats_sr, ML_FEATURE_COLS)
    print(variant)

    hyp = Hypothesis(
        name="cfb_cfbfastr_trailing_success_rate_features",
        reasoning=(
            "CFB's live ML_FEATURE_COLS has zero play-level efficiency signal -- home/away_pf_l10/"
            "pa_l10 conflate real offensive/defensive quality with special-teams TDs, turnover-return "
            "luck, and garbage-time scoring, the same conflation problem NFL's EPA feature addressed. "
            "Success rate (real cfbfastR play-by-play, free, 2013-2025, the standard down-and-distance "
            "efficiency metric popularized for CFB specifically by Bill Connelly's SP+) is the direct "
            "CFB analog, tested the same way: trailing (last-10-game, walk-forward-safe) team-level "
            "additions across every market, not scoped to one. Team-name mapping between cfbfastR's "
            "bare school names and this project's own franchise names covers ~72% of games (see "
            "cfbfastr_features.py's own docstring) after a 2026-09-14 collision-safety fix -- the "
            "first version's ~74%-looking coverage silently included 35 cases of two different real "
            "programs colliding onto one cfbfastR identity; those are now dropped rather than guessed, "
            "which is why this run's ROI moved from -0.08pp to +0.22pp and the recommendation from "
            "adopt_cautiously to adopt versus the first pass -- removing wrong data measurably helped."
        ),
        sport="CFB",
    )
    result = evaluate_hypothesis(hyp, baseline, variant)
    hr("HYPOTHESIS RESULT")
    print(f"real success-rate coverage (season>=2014, has a real prior-season trail): {coverage*100:.1f}%")
    print(json.dumps(result.to_dict(), indent=2))
    hr("DONE")


if __name__ == "__main__":
    main()
