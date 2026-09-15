"""
THROWAWAY research script -- NOT part of the production pipeline. Tests
the trailing EPA/success-rate features built from real nflverse play-by-
play data (sports/nfl/nflfastr_features.py) against NFL's current
production ML_FEATURE_COLS, via this project's standard Hypothesis
framework -- same discipline as every other feature test here (real,
externally-motivated reasoning, walk-forward, statistical-fit check
alongside ROI, never adopted on ROI alone).

MOTIVATION (app owner, 2026-09-14): "we have way too much data to not
have more confidence in what we are doing" -- a fair challenge on the
handful of markets stuck at Unvalidated for lack of a working confidence
signal (NOT the ones stuck there for lack of real market data, like CFB
spread -- see core/edge_finder.py's own comments for that distinction).
NFL's own feature set has zero play-level efficiency signal today --
team_pf_l10/pa_l10 are just points scored/allowed, which conflate real
offensive/defensive quality with special-teams TDs, turnover luck, and
garbage-time scoring. EPA/play and success rate are the standard modern
answer to exactly that conflation problem -- real, externally-motivated,
not "let's throw more numbers at it."

Run with:  python -m sports.nfl.research_nflfastr_epa_features   (from backend/)
Requires Datasets/NFL/nflfastr_team_game_epa.csv to exist first -- run
scripts/fetch_nflfastr_team_game_epa.py once if it doesn't.
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
from sports.nfl.nflfastr_features import build_trailing_epa_features, TRAILING_COLS


def hr(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def run_pipeline(feats, feature_cols) -> dict:
    wf = walk_forward_predict(feats, feature_cols)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])

    stds = ensemble.compute_residual_stds(oos, nfl_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()
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
    hr("BUILDING BASE PIPELINE (current production state)")
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
    print(f"Base feature rows: {len(feats_base):,}, base ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}")

    hr("BASELINE RUN (current production ML_FEATURE_COLS, every market)")
    baseline = run_pipeline(feats_base, ML_FEATURE_COLS)
    print(baseline)

    hr("HYPOTHESIS: adding real nflverse trailing EPA/success-rate features")
    feats_epa = build_trailing_epa_features(feats_base, n_games=10)
    epa_cols = [f"{side}_{col}_trail" for side in ("home", "away") for col in TRAILING_COLS]
    coverage = (feats_epa["season"] >= 2007).mean()  # first real season is 2006, so 2007+ has a real prior-season trail
    variant_cols = ML_FEATURE_COLS + epa_cols
    variant = run_pipeline(feats_epa, variant_cols)
    print(variant)

    hyp = Hypothesis(
        name="nfl_nflverse_trailing_epa_features",
        reasoning=(
            "NFL's live ML_FEATURE_COLS has zero play-level efficiency signal -- team_pf_l10/pa_l10 "
            "conflate real offensive/defensive quality with special-teams TDs, turnover-return "
            "luck, and garbage-time scoring. EPA/play and success rate (real nflverse play-by-play, "
            "free/MIT-licensed, 2006-2025 matching this project's own historical window exactly) are "
            "the standard modern answer to that conflation, externally validated across the real "
            "analytics community, not invented for this test. Tested as trailing (last-10-game, "
            "walk-forward-safe) team-level additions across every market, not scoped to one."
        ),
        sport="NFL",
    )
    result = evaluate_hypothesis(hyp, baseline, variant)
    hr("HYPOTHESIS RESULT")
    print(f"real EPA coverage (season>=2007, has a real prior-season trail): {coverage*100:.1f}%")
    print(json.dumps(result.to_dict(), indent=2))
    hr("DONE")


if __name__ == "__main__":
    main()
