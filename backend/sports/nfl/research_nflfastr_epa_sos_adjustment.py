"""
THROWAWAY research script -- tests the opponent-adjusted (strength-of-
schedule) trailing EPA/success-rate features
(sports/nfl/nflfastr_features.build_trailing_epa_features_sos_adjusted)
ADDED ON TOP of the already-adopted raw trailing EPA features, via this
project's standard Hypothesis framework.

MOTIVATION: direct answer to the "ground-up model design" question raised
this session (app owner, 2026-09-14) -- "opponent-adjustment is the
biggest gap" in the raw trailing EPA/success-rate features adopted
earlier this session: they're schedule-BLIND (three good games against
bad defenses looks identical to three good games against good ones),
unlike Elo, which already opponent-adjusts by construction. This tests
whether a real SOS adjustment (see that function's own docstring for the
method) carries INCREMENTAL signal beyond raw trailing EPA + Elo, not
whether it beats raw trailing EPA alone (raw stays in ML_FEATURE_COLS
either way -- this only tests adding the SOS-adjusted columns on top).

Includes a REQUIRED split-half stability check, not just the full-sample
evaluate_hypothesis() result -- see sports/nfl/research_qb_features.py's
own docstring/decision_log.jsonl entry for exactly why a full-sample-only
"adopt" is not trustworthy on its own (a real full-sample "adopt" from
that same session did not survive this exact check).

Run:  python -m sports.nfl.research_nflfastr_epa_sos_adjustment   (from backend/)
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
from sports.nfl.nflfastr_features import (
    build_trailing_epa_features, build_trailing_epa_features_sos_adjusted, TRAILING_COLS,
)


def hr(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def run_pipeline(feats, feature_cols) -> dict:
    wf = walk_forward_predict(feats, feature_cols)
    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    margin_corr = float(np.corrcoef(oos["predicted_margin"], oos["actual_margin"])[0, 1])
    total_corr = float(np.corrcoef(oos["predicted_total"], oos["actual_total"])[0, 1])

    stds = ensemble.compute_residual_stds(oos, nfl_config.ELO_POINTS_PER_MARGIN)
    ens_cfg = ensemble.EnsembleConfig()  # plain default, isolates this test's own effect from the spread-weight adoption
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


def build_all_feats():
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
    feats = build_features(games, rr.history)
    feats = build_trailing_epa_features(feats)  # current production baseline (raw trailing EPA, already adopted)
    feats = build_trailing_epa_features_sos_adjusted(feats)  # adds the _trail_sos columns on top
    return feats


def main():
    hr("BUILDING FEATURES (raw trailing EPA + SOS-adjusted trailing EPA, both present)")
    feats = build_all_feats()
    sos_cols = [f"{side}_{c}_trail_sos" for side in ("home", "away") for c in TRAILING_COLS]
    print(f"Feature rows: {len(feats):,}, base ML_FEATURE_COLS: {len(ML_FEATURE_COLS)}, SOS cols: {len(sos_cols)}")

    hr("BASELINE RUN (current production ML_FEATURE_COLS -- raw trailing EPA, no SOS adjustment)")
    baseline = run_pipeline(feats, ML_FEATURE_COLS)
    print(baseline)

    hr("HYPOTHESIS: adding SOS-adjusted trailing EPA/success-rate ON TOP of raw")
    variant_cols = ML_FEATURE_COLS + sos_cols
    variant = run_pipeline(feats, variant_cols)
    print(variant)

    hyp = Hypothesis(
        name="nfl_nflverse_trailing_epa_sos_adjustment",
        reasoning=(
            "Raw trailing EPA/success-rate (adopted this session) is schedule-BLIND -- unlike Elo, "
            "which already opponent-adjusts by construction, three strong offensive games against weak "
            "defenses look identical to three strong games against elite ones. This tests a real "
            "opponent adjustment (subtract the opponent's own trailing allowed-rate at the time of each "
            "past game, re-center to league average, then roll the adjusted per-game values -- the same "
            "idea SRS-style ratings use) added ON TOP of the raw trailing features, not replacing them, "
            "to see if it carries incremental signal beyond raw trailing EPA + Elo combined."
        ),
        sport="NFL",
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
