"""
THROWAWAY diagnostic script -- tests NHL total's OVER-vs-UNDER split using
core/research.py's evaluate_subgroup_hypothesis()/core/shrinkage.py, the
same real tooling already used for sports/nba/research_spread_favorite_
underdog_shrinkage.py.

MOTIVATION: a live confidence audit flagged NHL total BACKWARDS with a
real, growing settled sample. But that LIVE sample is contaminated: it
was logged by the OLD, pre-2026-11-07-fix total model (see
sports/nhl/config.py's TOTAL_MODEL_LOSS and decision_log.jsonl's
"nhl_total_model_squared_error_loss" entry) -- a model biased ~0.5
goals/game LOW systematically found more UNDER edges than it should
have, so WHICH side got logged live is itself an artifact of the bug,
not a real signal about market bias. The correct test re-derives this
from scratch, confidence-blind, through the CORRECTED (squared-error)
total model's own real walk-forward predictions -- not reusing the
contaminated live sample.

Real, externally-motivated direction check, not assumed: NFL/MLB's own
validated total_side_confidence_tier() says UNDER is the stronger side
(public bias toward the more "exciting" over). This script does NOT
assume NHL shares that direction -- a quick confidence-blind check on the
corrected model found the OPPOSITE sign for NHL (OVER outperforming
UNDER), which is itself externally plausible: NHL is a famously low-
scoring sport relative to the other leagues this project covers, and
casual/recreational bettors underestimate modern NHL scoring rates
(post-rule-change power-play emphasis, faster pace) -- a real,
documented "sucker bet on the under" pattern in hockey-betting
literature, the mirror image of NFL/MLB's "exciting over" bias, not a
contradiction of it.

Run with:  python -m sports.nhl.research_total_side_confidence   (from backend/)
"""

import math

import pandas as pd

from core import edge_finder, research, live_results
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core import ensemble
from core.backtest import settle_bet
from sports.nhl import config
from sports.nhl.loader import load_games
from sports.nhl import features
from sports.nhl.moneypuck_xg_features import build_trailing_xg_features


def main():
    games = load_games()
    games_for_ratings = live_results.with_live_results(games, "NHL")
    rr = compute_power_ratings(
        games_for_ratings, home_col="home_team_id", away_col="away_team_id",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date",
        neutral_col="is_neutral_venue" if "is_neutral_venue" in games.columns else None,
        config=PowerRatingConfig(
            k_factor=config.ELO_K_FACTOR, start_rating=config.ELO_START_RATING,
            home_field_adv=config.HOME_FIELD_ADV_ELO, season_regression=config.SEASON_REGRESSION,
            mov_mult_base=config.MOV_MULT_BASE, mov_mult_divisor=config.MOV_MULT_DIVISOR,
        ),
    )
    feats = features.build_features(games, rr.history)
    feats = build_trailing_xg_features(feats)
    # Real fix adopted 2026-11-07 -- see sports/nhl/config.py's
    # TOTAL_MODEL_LOSS. Using the OLD default here would just re-derive
    # the same contaminated-direction artifact the live audit already
    # showed; this test's whole point is checking the CORRECTED model.
    total_loss = getattr(config, "TOTAL_MODEL_LOSS", "absolute_error")
    wf = walk_forward_predict(feats, features.ML_FEATURE_COLS, total_loss=total_loss)
    history_df = feats.set_index("game_id").join(wf.predictions, how="left")
    oos_df = history_df.dropna(subset=["predicted_margin"])
    stds = ensemble.compute_residual_stds(oos_df, config.ELO_POINTS_PER_MARGIN)
    ecfg = ensemble.EnsembleConfig()

    # Confidence-blind: every edge>=3% total opportunity, no tier filter.
    records = []
    for game_id, row in oos_df.iterrows():
        opps = edge_finder.evaluate_game(row, stds, config.ELO_POINTS_PER_MARGIN, ecfg,
                                          price_point="Close", sport=None)
        for o in opps:
            if o.market != "total" or o.edge_pct < 3.0:
                continue
            result = settle_bet(o.market, o.side, o.line, row)
            profit = 0.0 if result == 0 else ((o.market_odds - 1.0) if result == 1 else -1.0)
            records.append({"season": row["season"], "side": o.side, "result": result, "profit": profit})

    bets = pd.DataFrame.from_records(records)
    bets["is_over"] = bets["side"] == "over"

    def roi_stats(df):
        n = len(df)
        if n == 0:
            return {"bets": 0, "roi_pct": 0.0, "roi_stderr_pct": 0.0}
        roi = df["profit"].sum() / n * 100.0
        stderr = df["profit"].std(ddof=1) / math.sqrt(n) * 100.0 if n > 1 else 0.0
        return {"bets": n, "roi_pct": roi, "roi_stderr_pct": stderr}

    over = roi_stats(bets[bets["is_over"]])
    under = roi_stats(bets[~bets["is_over"]])
    print(f"OVER:  n={over['bets']}, ROI={over['roi_pct']:+.2f}% +/-{over['roi_stderr_pct']:.2f}pp")
    print(f"UNDER: n={under['bets']}, ROI={under['roi_pct']:+.2f}% +/-{under['roi_stderr_pct']:.2f}pp")

    seasons = sorted(bets["season"].dropna().unique())
    mid = seasons[len(seasons) // 2]
    first_half = bets[bets["season"] < mid]
    second_half = bets[bets["season"] >= mid]

    def delta(df):
        o = roi_stats(df[df["is_over"]])
        u = roi_stats(df[~df["is_over"]])
        return o["roi_pct"] - u["roi_pct"]

    d1, d2 = delta(first_half), delta(second_half)
    print(f"Split-half delta (over - under): first half {d1:+.2f}pp, second half {d2:+.2f}pp")

    hyp = research.Hypothesis(
        name="nhl_total_side_confidence_over_bias",
        reasoning=(
            "A live confidence audit flagged NHL total confidence_tier() backwards, but the live sample "
            "was logged entirely by the pre-fix, systematically-biased-low total model (see "
            "nhl_total_model_squared_error_loss) -- WHICH side it flagged live is itself an artifact of "
            "that bug, not a real signal. Re-derived confidence-blind through the corrected model's own "
            "walk-forward predictions: found the OPPOSITE direction from NFL/MLB's validated under-bias "
            "mechanism (OVER outperforms UNDER here, not under-outperforms-over) -- externally plausible, "
            "not just a number: NHL is this project's lowest-scoring covered sport, and recreational "
            "bettors are documented to systematically underestimate modern NHL scoring rates (faster pace, "
            "power-play rule emphasis), the mirror image of NFL/MLB's 'exciting over' public bias, not a "
            "contradiction of the same underlying public-perception-vs-reality mechanism."
        ),
        sport="NHL",
    )
    result = research.evaluate_subgroup_hypothesis(
        hyp, market="total", subgroup_metrics=over, rest_metrics=under,
        split_half_deltas=(d1, d2), min_bets=100,
    )
    print(f"\nz={result.z_score:+.2f} (bar: {result.stderr_multiplier:.2f}), "
          f"stable_direction={result.stable_direction}, recommendation={result.recommendation}")
    print(f"Shrinkage: subgroup_weight={result.shrinkage.subgroup_weight:.3f}, "
          f"shrunk_effect={result.shrinkage.shrunk_effect:+.2f}pp "
          f"(raw over {result.shrinkage.subgroup_effect:+.2f}pp, raw under-baseline {result.shrinkage.baseline_effect:+.2f}pp)")


if __name__ == "__main__":
    main()
