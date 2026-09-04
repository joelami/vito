"""
THROWAWAY diagnostic script -- first real, worked application of
core/research.py's new evaluate_subgroup_hypothesis() / core/shrinkage.py.

Re-derives NBA spread's favorite-vs-underdog gap (already found and
honestly left unresolved in the 2026-09-03 "4 remaining backwards cases"
investigation: favorite +2.00% ROI n=2069 vs underdog -1.16% ROI n=9463,
delta 3.16pp on ~2.36pp combined stderr, z~1.34, stable direction across
both season halves) DIRECTLY from the true confidence-blind population
(every edge>=3% opportunity via edge_finder.evaluate_game(sport=None), no
confidence-tier filter) -- not reusing the old script's printed numbers,
per this project's own re-derive-before-trusting discipline.

This is a DIAGNOSTIC, not a hypothesis about a NEW feature -- nothing here
is proposed for adoption into ML_FEATURE_COLS. It exists to test the new
subgroup-shrinkage machinery against a real, already-understood case with
a known expected outcome (z~1.34 -- real direction, not yet significant),
and to produce the actual watchlist entry.

Run with:  python -m sports.nba.research_spread_favorite_underdog_shrinkage   (from backend/)
"""

import math

from core import edge_finder, research
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from core import ensemble
from sports.nba import config
from sports.nba.loader import load_games
from sports.nba import features


def main():
    games = load_games()
    games_for_ratings = games
    rr = compute_power_ratings(
        games_for_ratings, home_col="home_franchise", away_col="away_franchise",
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
    wf = walk_forward_predict(feats, features.ML_FEATURE_COLS)
    history_df = feats.set_index("game_id").join(wf.predictions, how="left")
    oos_df = history_df.dropna(subset=["predicted_margin"])
    stds = ensemble.compute_residual_stds(oos_df, config.ELO_POINTS_PER_MARGIN)
    ecfg = ensemble.EnsembleConfig()

    # Confidence-blind: every edge>=3% spread opportunity, no tier filter.
    records = []
    for game_id, row in oos_df.iterrows():
        opps = edge_finder.evaluate_game(row, stds, config.ELO_POINTS_PER_MARGIN, ecfg,
                                          price_point="Close", sport=None)
        for o in opps:
            if o.market != "spread" or o.edge_pct < 3.0:
                continue
            from core.backtest import settle_bet, CLOSE_ODDS_COL
            result = settle_bet(o.market, o.side, o.line, row)
            profit = 0.0 if result == 0 else ((o.market_odds - 1.0) if result == 1 else -1.0)
            records.append({
                "season": row["season"], "line": o.line, "side": o.side,
                "result": result, "profit": profit,
            })

    import pandas as pd
    bets = pd.DataFrame.from_records(records)
    # favorite = negative line for the side actually bet (home favorite bets
    # home with line<0, away favorite bets away with line<0 from that side's
    # perspective) -- edge_finder stores `line` as the number for the side
    # actually bet, so line<0 IS "betting the favorite" regardless of home/away.
    bets["is_favorite"] = bets["line"] < 0

    def roi_stats(df):
        n = len(df)
        roi = df["profit"].sum() / n * 100.0
        stderr = df["profit"].std(ddof=1) / math.sqrt(n) * 100.0
        return {"bets": n, "roi_pct": roi, "roi_stderr_pct": stderr}

    fav = roi_stats(bets[bets["is_favorite"]])
    dog = roi_stats(bets[~bets["is_favorite"]])
    print(f"Favorite: n={fav['bets']}, ROI={fav['roi_pct']:+.2f}% +/-{fav['roi_stderr_pct']:.2f}pp")
    print(f"Underdog: n={dog['bets']}, ROI={dog['roi_pct']:+.2f}% +/-{dog['roi_stderr_pct']:.2f}pp")

    seasons = sorted(bets["season"].dropna().unique())
    mid = seasons[len(seasons) // 2]
    first_half = bets[bets["season"] < mid]
    second_half = bets[bets["season"] >= mid]

    def delta(df):
        f = roi_stats(df[df["is_favorite"]])
        d = roi_stats(df[~df["is_favorite"]])
        return f["roi_pct"] - d["roi_pct"]

    d1, d2 = delta(first_half), delta(second_half)
    print(f"Split-half delta (favorite - underdog): first half {d1:+.2f}pp, second half {d2:+.2f}pp")

    hyp = research.Hypothesis(
        name="nba_spread_favorite_underdog_subgroup",
        reasoning="NBA spread's standing confidence audit shows a real, stable-direction ROI gap between "
                  "betting the favorite (negative line) vs. the underdog on spread bets, motivated by the "
                  "well-documented public-money bias toward popular/star-driven underdog narratives in NBA "
                  "betting specifically -- but the gap (~3.16pp on ~2.36pp combined stderr, z~1.34) has never "
                  "cleared this project's adoption bar in 3 separate re-checks. Testing whether the new "
                  "shrinkage-based subgroup framework gives an honest, gradual answer instead of a forced "
                  "binary adopt/reject.",
        sport="NBA",
    )
    result = research.evaluate_subgroup_hypothesis(
        hyp, market="spread", subgroup_metrics=fav, rest_metrics=dog,
        split_half_deltas=(d1, d2), min_bets=100,
    )
    print(f"\nz={result.z_score:+.2f} (bar: {result.stderr_multiplier:.2f}), "
          f"stable_direction={result.stable_direction}, recommendation={result.recommendation}")
    print(f"Shrinkage: subgroup_weight={result.shrinkage.subgroup_weight:.3f}, "
          f"shrunk_effect={result.shrinkage.shrunk_effect:+.2f}pp "
          f"(raw favorite {result.shrinkage.subgroup_effect:+.2f}pp, raw underdog-baseline {result.shrinkage.baseline_effect:+.2f}pp)")


if __name__ == "__main__":
    main()
