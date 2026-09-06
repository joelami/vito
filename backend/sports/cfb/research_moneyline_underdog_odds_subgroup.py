"""
THROWAWAY diagnostic script -- second, complementary fix for the CFB
moneyline problem flagged 2026-09-05 (see decision_log.jsonl and sports/cfb/
research_rare_opponent_start_rating.py, already adopted). That fix gave
FCS/small-conference "buy game" opponents a real rating floor instead of
the same flat 1500 start as an established program, and it measurably
improved the model's fit -- but it did NOT fully fix the most extreme
underdog-price bets on its own: the market_odds>=10 bucket went from
0/44 (-100% ROI) to 2/52 (-55.3% ROI) under the corrected ratings -- less
catastrophic, still a real loser.

This tests whether that remaining tail clears this project's real adoption
bar as its OWN subgroup finding, using core/research.py's
evaluate_subgroup_hypothesis() (the same shrinkage + season-split-half-
stability machinery sports/nba/research_spread_favorite_underdog_shrinkage.py
introduced) -- subgroup = CFB moneyline bets at market_odds>=10 (a real,
extreme underdog price), rest = every other CFB moneyline bet clearing the
same 3% edge bar. Confidence-blind (sport=None in evaluate_game, like the
NBA template) since this subgroup axis (odds magnitude) is independent of
confidence tier -- testing whether IT is real, not re-deriving confidence
tier's own finding.

Run with:  python -m sports.cfb.research_moneyline_underdog_odds_subgroup   (from backend/)
"""

import math

import pandas as pd

from core import edge_finder, research, live_results
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core import ensemble
from core.backtest import settle_bet
from sports.cfb import config
from sports.cfb.loader import load_games
from sports.cfb import features

ODDS_THRESHOLD = 10.0  # matches the real, already-identified worst bucket


def main():
    games = load_games()
    games_for_ratings = live_results.with_live_results(games, "CFB")
    rr = compute_power_ratings(
        games_for_ratings, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date",
        neutral_col="is_neutral_venue" if "is_neutral_venue" in games.columns else None,
        config=PowerRatingConfig(
            k_factor=config.ELO_K_FACTOR, start_rating=config.ELO_START_RATING,
            home_field_adv=config.HOME_FIELD_ADV_ELO, season_regression=config.SEASON_REGRESSION,
            mov_mult_base=config.MOV_MULT_BASE, mov_mult_divisor=config.MOV_MULT_DIVISOR,
            # CURRENT production state (adopted 2026-09-05) -- this subgroup
            # test needs to run against the ratings the live app actually
            # uses now, not the pre-fix ones, since that's the real
            # remaining tail this is testing.
            low_history_start_rating=config.LOW_HISTORY_START_RATING,
            low_history_game_threshold=config.LOW_HISTORY_GAME_THRESHOLD,
        ),
    )
    feats = features.build_features(games, rr.history)
    wf = walk_forward_predict(feats, features.ML_FEATURE_COLS)
    history_df = feats.set_index("game_id").join(wf.predictions, how="left")
    oos_df = history_df.dropna(subset=["predicted_margin"])
    stds = ensemble.compute_residual_stds(oos_df, config.ELO_POINTS_PER_MARGIN)
    ecfg = ensemble.EnsembleConfig()

    # Confidence-blind: every edge>=3% CFB moneyline opportunity, no tier filter.
    records = []
    for game_id, row in oos_df.iterrows():
        opps = edge_finder.evaluate_game(row, stds, config.ELO_POINTS_PER_MARGIN, ecfg,
                                          price_point="Close", sport=None)
        for o in opps:
            if o.market != "moneyline" or o.edge_pct < 3.0:
                continue
            result = settle_bet(o.market, o.side, o.line, row)
            profit = 0.0 if result == 0 else ((o.market_odds - 1.0) if result == 1 else -1.0)
            records.append({
                "season": row["season"], "market_odds": o.market_odds,
                "result": result, "profit": profit,
            })

    bets = pd.DataFrame.from_records(records)
    bets["is_extreme_dog"] = bets["market_odds"] >= ODDS_THRESHOLD

    def roi_stats(df):
        n = len(df)
        if n == 0:
            return {"bets": 0, "roi_pct": float("nan"), "roi_stderr_pct": float("nan")}
        roi = df["profit"].sum() / n * 100.0
        stderr = df["profit"].std(ddof=1) / math.sqrt(n) * 100.0 if n > 1 else float("nan")
        return {"bets": n, "roi_pct": roi, "roi_stderr_pct": stderr}

    extreme = roi_stats(bets[bets["is_extreme_dog"]])
    rest = roi_stats(bets[~bets["is_extreme_dog"]])
    print(f"Extreme underdog (odds>={ODDS_THRESHOLD}): n={extreme['bets']}, "
          f"ROI={extreme['roi_pct']:+.2f}% +/-{extreme['roi_stderr_pct']:.2f}pp")
    print(f"Rest of CFB moneyline: n={rest['bets']}, "
          f"ROI={rest['roi_pct']:+.2f}% +/-{rest['roi_stderr_pct']:.2f}pp")

    seasons = sorted(bets["season"].dropna().unique())
    mid = seasons[len(seasons) // 2]
    first_half = bets[bets["season"] < mid]
    second_half = bets[bets["season"] >= mid]

    def delta(df):
        e = roi_stats(df[df["is_extreme_dog"]])
        r = roi_stats(df[~df["is_extreme_dog"]])
        if e["bets"] == 0 or r["bets"] == 0:
            return float("nan")
        return e["roi_pct"] - r["roi_pct"]

    d1, d2 = delta(first_half), delta(second_half)
    print(f"Split-half delta (extreme underdog - rest): first half {d1:+.2f}pp "
          f"(seasons < {mid}, n={len(first_half)}), second half {d2:+.2f}pp "
          f"(seasons >= {mid}, n={len(second_half)})")

    hyp = research.Hypothesis(
        name="cfb_moneyline_extreme_underdog_odds_subgroup",
        reasoning=(
            f"Post-rating-floor-fix, CFB moneyline's market_odds>={ODDS_THRESHOLD} bucket is still "
            f"the single worst-performing slice (was 0/44 -100% ROI pre-fix, 2/52 -55.3% ROI "
            f"post-fix -- improved but still a real loser), unlike the moderate-underdog buckets "
            f"which the rating fix visibly helped. Testing whether a hard exclusion at this "
            f"threshold clears this project's real adoption bar as a standalone subgroup finding, "
            f"stable across a season split-half, not just a artifact of the pooled sample."
        ),
        sport="CFB",
    )
    result = research.evaluate_subgroup_hypothesis(
        hyp, market="moneyline", subgroup_metrics=extreme, rest_metrics=rest,
        split_half_deltas=(d1, d2), min_bets=30,
    )
    print(f"\nz={result.z_score:+.2f} (bar: {result.stderr_multiplier:.2f}), "
          f"stable_direction={result.stable_direction}, recommendation={result.recommendation}")
    print(f"Shrinkage: subgroup_weight={result.shrinkage.subgroup_weight:.3f}, "
          f"shrunk_effect={result.shrinkage.shrunk_effect:+.2f}pp "
          f"(raw extreme-dog {result.shrinkage.subgroup_effect:+.2f}pp, raw rest {result.shrinkage.baseline_effect:+.2f}pp)")


if __name__ == "__main__":
    main()
