"""
THROWAWAY research script -- tests the opponent-adjusted (strength-of-
schedule) trailing success-rate features
(sports/cfb/cfbfastr_features.build_trailing_success_features_sos_adjusted)
ADDED ON TOP of the already-adopted raw trailing success-rate features, via
this project's standard Hypothesis framework. Direct CFB transfer of NFL's
nfl_nflverse_trailing_epa_sos_adjustment (adopted 2026-09-15, see
decision_log.jsonl) -- flagged as the cleanest cross-sport transfer
candidate in docs/model_improvement_candidates_2026-09-15.md's "Cross-sport
structural ideas" section: CFB's raw off_success_rate_trail/
def_success_rate_allowed_trail are the exact same shape NFL's raw trailing
EPA was BEFORE its own SOS-adjustment, and CFB's real cross-conference
strength disparity (SEC vs. Sun Belt, say) is exactly the case this method
is built for.

MOTIVATION: raw trailing success rate is schedule-BLIND -- three good
offensive games against weak defenses look identical to three good games
against elite ones, even though the second is a much stronger real signal
of offensive quality. This tests whether a real SOS adjustment (same
method as NFL's, see cfbfastr_features.py's module comment for the CFB-
specific coverage-gap handling) carries INCREMENTAL signal beyond raw
trailing success rate + Elo, not whether it beats raw trailing success
rate alone (raw stays in ML_FEATURE_COLS either way -- this only tests
adding the SOS-adjusted columns on top).

Includes a REQUIRED split-half stability check, not just the full-sample
evaluate_hypothesis() result -- see sports/nfl/research_qb_features.py's
own docstring/decision_log.jsonl entry for exactly why a full-sample-only
"adopt" is not trustworthy on its own (a real full-sample "adopt" from that
same NFL session did not survive this exact check), and
sports/nfl/research_nflfastr_epa_sos_adjustment.py for the reference
implementation of this exact check being applied to the SOS-adjustment
method specifically.

Uses the GENERIC build_pipeline() shape's own logic (see pipeline.py's
build_pipeline, NOT build_nfl_pipeline -- CFB has no dedicated pipeline
function), matching research_cfbfastr_success_features.py's own pattern.

Run:  python -m sports.cfb.research_success_rate_sos_adjustment   (from backend/)
Requires Datasets/College Football/cfbfastr_team_game_success.csv to exist
first -- run scripts/fetch_cfbfastr_team_game_success.py once if it doesn't.
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
from sports.cfb.cfbfastr_features import (
    build_trailing_success_features, build_trailing_success_features_sos_adjusted, TRAILING_COLS,
)


def hr(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def run_pipeline(feats, feature_cols) -> dict:
    wf = walk_forward_predict(feats, feature_cols)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])

    stds = ensemble.compute_residual_stds(oos, cfb_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()  # plain default, isolates this test's own effect
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


def build_all_feats():
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
    feats = build_features(games, rr.history)
    feats = build_trailing_success_features(feats)  # current production baseline (raw trailing success rate, already adopted)
    feats = build_trailing_success_features_sos_adjusted(feats)  # adds the _trail_sos columns on top
    return feats


def main():
    hr("BUILDING FEATURES (raw trailing success rate + SOS-adjusted trailing success rate, both present)")
    feats = build_all_feats()
    sos_cols = [f"{side}_{c}_trail_sos" for side in ("home", "away") for c in TRAILING_COLS]
    print(f"Feature rows: {len(feats):,}, base ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}, SOS cols: {len(sos_cols)}")

    hr("BASELINE RUN (current production ML_FEATURE_COLS -- raw trailing success rate, no SOS adjustment)")
    baseline = run_pipeline(feats, ML_FEATURE_COLS)
    print(baseline)

    hr("HYPOTHESIS: adding SOS-adjusted trailing success rate ON TOP of raw")
    variant_cols = ML_FEATURE_COLS + sos_cols
    variant = run_pipeline(feats, variant_cols)
    print(variant)

    hyp = Hypothesis(
        name="cfb_cfbfastr_success_rate_sos_adjustment",
        reasoning=(
            "CFB's already-adopted raw trailing success rate (off_success_rate_trail/"
            "def_success_rate_allowed_trail) is schedule-BLIND, the exact same shape NFL's raw trailing "
            "EPA was BEFORE its own SOS-adjustment (nfl_nflverse_trailing_epa_sos_adjustment, adopted "
            "this same project this session) -- three strong offensive games against weak defenses look "
            "identical to three strong games against elite ones. CFB's real cross-conference strength "
            "disparity (SEC vs. Sun Belt, say) is exactly the case this method is built for, and is "
            "flagged as the cleanest transfer candidate in docs/model_improvement_candidates_2026-09-15.md's "
            "'Cross-sport structural ideas' section (same 'modern, play-level' quality tier NFL's raw "
            "trailing EPA was). This tests a direct transfer of NFL's opponent adjustment (subtract the "
            "opponent's own trailing allowed-rate at the time of each past game, re-center to league "
            "average, then roll the adjusted per-game values -- the same idea SRS-style ratings use) "
            "added ON TOP of the raw trailing features, not replacing them, to see if it carries "
            "incremental signal beyond raw trailing success rate + Elo combined. CFB's own real, honest "
            "team-name-mapping coverage limit (~72% of games, see cfbfastr_features.py's docstring) "
            "carries through to the opponent lookup here too -- an unmapped opponent falls back to a "
            "no-op (league-average) adjustment for that one game rather than dropping the team's own "
            "game from its trailing window."
        ),
        sport="CFB",
    )
    result = evaluate_hypothesis(hyp, baseline, variant)
    hr("HYPOTHESIS RESULT (full sample)")
    print(json.dumps(result.to_dict(), indent=2))

    hr("SPLIT-HALF STABILITY CHECK -- required before trusting the full-sample result above")
    seasons = sorted(feats["season"].unique())
    mid = seasons[len(seasons) // 2]
    stable = True
    for label, sub in [("first half", feats[feats["season"] < mid]),
                        ("second half", feats[feats["season"] >= mid])]:
        b = run_pipeline(sub, ML_FEATURE_COLS)
        v = run_pipeline(sub, variant_cols)
        md, td = v["margin_corr"] - b["margin_corr"], v["total_corr"] - b["total_corr"]
        print(f"{label}: margin {b['margin_corr']:.4f}->{v['margin_corr']:.4f} ({md:+.4f}), "
              f"total {b['total_corr']:.4f}->{v['total_corr']:.4f} ({td:+.4f})")
        if md < -0.005 or td < -0.005:
            stable = False
    print(f"\n{'STABLE' if stable else 'NOT STABLE'} across both halves (neither corr degraded past the noise floor "
          f"in either half) -- {'trust' if stable else 'do NOT trust'} the full-sample recommendation above.")
    hr("DONE")


if __name__ == "__main__":
    main()
