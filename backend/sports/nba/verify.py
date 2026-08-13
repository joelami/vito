"""
Standalone Phase-1 verification script for the NBA module. Runs the exact
same pipeline shape used for NFL/CFB (loader -> power ratings -> features ->
walk-forward ML -> residual stds -> backtest -> summarize) and prints every
number needed to judge whether this dataset supports a credible NBA model.
Not wired into main.py / the live app - backend-only validation pass.

UNLIKE MLB (zero odds data anywhere) and CFB (odds present but only assumed
-110 juice on spread/total, real two-sided pricing only on the moneyline),
NBA has REAL two-sided American-odds pricing on all three markets
(spread/total/moneyline) from Pinnacle Sports, 2006-07 through 2017-18. This
script runs a genuine backtest, not a residual-std-only check.

CLV caveat, verified directly (see loader.py docstring): the betting source
has exactly ONE price snapshot per (book, game) - no open/close/min/max like
the NFL source. `run_backtest` prices every bet at `price_point="Close"`
(the only price point that exists) and separately looks up the SAME
hardcoded "...Close" columns for its CLV calculation - since both point at
the identical price here, CLV is trivially ~0 for every bet, same honest
caveat as sports/cfb/verify.py. This backtest can only speak to hit-rate/ROI
against Pinnacle's one price point, not genuine closing-line value.

Run with:  python -m sports.nba.verify   (from backend/)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # backend/ on path
warnings.filterwarnings("ignore", category=FutureWarning)

from core import ensemble, edge_finder, backtest
from core.power_ratings import compute_power_ratings, PowerRatingConfig
from core.ml_models import walk_forward_predict
from sports.nba import config as nba_config
from sports.nba.loader import load_games
from sports.nba.features import build_features, ML_FEATURE_COLS


pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    hr("1. LOAD")
    games = load_games()
    print(f"Rows after load/clean (real teams, real dates, RS+Playoffs, valid scores): {len(games):,}")
    print(f"Season range: {games['season'].min()} - {games['season'].max()}")
    print(f"Unique home franchises: {games['home_franchise'].nunique()}  "
          f"(expect 30 - see config.py's team-identity verification)")

    has_spread = games["Home Line Close"].notna()
    has_total = games["Total Score Close"].notna()
    has_ml = games["Home Odds Close"].notna() & games["Away Odds Close"].notna()
    print(f"\nGames with a Pinnacle spread:    {has_spread.sum():,} ({has_spread.mean()*100:.1f}%)")
    print(f"Games with a Pinnacle total:     {has_total.sum():,} ({has_total.mean()*100:.1f}%)")
    print(f"Games with Pinnacle moneylines:  {has_ml.sum():,} ({has_ml.mean()*100:.1f}%)")
    print(f"Games with ANY Pinnacle odds:    {(has_spread | has_total | has_ml).sum():,} "
          f"({(has_spread | has_total | has_ml).mean()*100:.1f}%)")
    print("(Percentages are of ALL games back to 1950 - betting data only exists "
          "2006-07 onward, see below for coverage restricted to that window.)")

    odds_era = games[games["season"] >= 2006]
    has_spread_e = odds_era["Home Line Close"].notna()
    print(f"\nRestricted to season >= 2006 ({len(odds_era):,} games): "
          f"spread coverage {has_spread_e.mean()*100:.1f}%")

    print("\nOdds coverage by season (spread/total/ML coverage %, seasons with any betting data):")
    cov = games.assign(has_spread=has_spread, has_total=has_total, has_ml=has_ml)
    cov = cov[cov["season"] >= 2006]
    cov_by_season = cov.groupby("season").agg(
        n=("game_id", "size"),
        spread_pct=("has_spread", "mean"),
        total_pct=("has_total", "mean"),
        ml_pct=("has_ml", "mean"),
    )
    print((cov_by_season * [1, 100, 100, 100]).round(1).to_string())

    # cross-check season_for_date against the CSV's own season_year column, on the RAW file
    raw = pd.read_csv(nba_config.GAMES_PATH, dtype={"game_id": str})
    raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce")
    raw = raw.dropna(subset=["game_date"])
    raw = raw[raw["season_type"].isin(nba_config.KEPT_SEASON_TYPES)]
    derived = raw["game_date"].apply(nba_config.season_for_date)
    mismatches = (derived != raw["season_year"]).sum()
    print(f"\nseason_for_date(date) vs CSV season_year column mismatches "
          f"(non-NaT dates, RS+Playoffs only): {mismatches} / {len(raw)}")

    print("\nCLV NOTE: this dataset has exactly one price snapshot per (book, game) - "
          "no true open vs close (verified: max rows per (game_id, book_name) across all "
          "three betting files = 1). See this module's docstring for the full caveat.")

    # ------------------------------------------------------------------
    # 2. Power ratings
    # ------------------------------------------------------------------
    hr("2. POWER RATINGS")
    rating_cfg = PowerRatingConfig(
        k_factor=nba_config.ELO_K_FACTOR,
        start_rating=nba_config.ELO_START_RATING,
        home_field_adv=nba_config.HOME_FIELD_ADV_ELO,
        season_regression=nba_config.SEASON_REGRESSION,
        mov_mult_base=nba_config.MOV_MULT_BASE,
        mov_mult_divisor=nba_config.MOV_MULT_DIVISOR,
    )
    print(f"Config: k={rating_cfg.k_factor}, hfa={rating_cfg.home_field_adv}, "
          f"season_regression={rating_cfg.season_regression}, "
          f"mov_base={rating_cfg.mov_mult_base}, mov_divisor={rating_cfg.mov_mult_divisor}")

    actual_home_win_rate = games[games["season"] >= 2006]["home_win"].mean()
    print(f"\nActual home win rate, odds-era games (season >= 2006): {actual_home_win_rate:.4f} "
          f"(the well-documented NBA figure is ~0.60; home_field_adv was selected by the "
          f"calibration-decile grid search, not solved directly against this number - see config.py)")

    rr = compute_power_ratings(
        games, home_col="home_franchise", away_col="away_franchise",
        home_score_col="home_score", away_score_col="away_score",
        season_col="season", date_col="date",
        config=rating_cfg,
    )
    hist = rr.history.join(games.set_index("game_id")[["home_win", "date", "season"]])

    print("\nCalibration check (elo_home_win_prob decile vs actual home-win rate), full history:")
    hist["decile"] = pd.qcut(hist["elo_home_win_prob"], 10, duplicates="drop")
    calib = hist.groupby("decile", observed=True).agg(
        n=("home_win", "size"),
        mean_pred=("elo_home_win_prob", "mean"),
        actual_home_win_rate=("home_win", "mean"),
    )
    print(calib.to_string(float_format=lambda x: f"{x:.4f}"))
    calib_err = (calib["mean_pred"] - calib["actual_home_win_rate"]).abs().mean()
    print(f"\nMean |predicted - actual| across deciles: {calib_err:.4f}")

    # ------------------------------------------------------------------
    # 3. Features
    # ------------------------------------------------------------------
    hr("3. FEATURES")
    feats = build_features(games, rr.history)
    print(f"Feature rows: {len(feats):,}, columns added: "
          f"{len([c for c in feats.columns if c not in games.columns])}")
    print(f"Back-to-back rate (home_b2b==1): {feats['home_b2b'].mean()*100:.1f}% of home games, "
          f"{feats['away_b2b'].mean()*100:.1f}% of away games - a real, NBA-specific schedule feature "
          f"not present in the other sports' modules (see features.py docstring).")

    # ------------------------------------------------------------------
    # 4. Walk-forward ML
    # ------------------------------------------------------------------
    hr("4. WALK-FORWARD ML")
    wf = walk_forward_predict(feats, ML_FEATURE_COLS, min_train_seasons=3)
    print(f"Seasons predicted out-of-sample: {wf.seasons_predicted}")
    print(f"OOS rows: {len(wf.predictions):,}")
    print(f"Margin residual std: {wf.margin_residual_std:.2f}")
    print(f"Total residual std:  {wf.total_residual_std:.2f}")

    oos = feats.set_index("game_id").join(wf.predictions, how="inner")

    def corr(a, b):
        return float(np.corrcoef(oos[a], oos[b])[0, 1])

    print(f"\nMargin: ML pred vs actual correlation:    {corr('predicted_margin', 'actual_margin'):.4f}")
    print(f"Margin: naive (rating_diff/elo_ppm) vs actual correlation: "
          f"{corr('rating_diff_pre', 'actual_margin'):.4f}  (rating_diff_pre, unscaled, for direction only)")
    elo_pred_margin = oos["rating_diff_pre"] / nba_config.ELO_POINTS_PER_MARGIN
    elo_margin_corr = float(np.corrcoef(elo_pred_margin, oos["actual_margin"])[0, 1])
    print(f"Margin: pure Elo-implied margin vs actual correlation: {elo_margin_corr:.4f}")
    print(f"Total:  ML pred vs actual correlation:    {corr('predicted_total', 'actual_total'):.4f}")
    print(f"Total:  naive pace baseline vs actual correlation: {corr('naive_total', 'actual_total'):.4f}")

    mae_ml_margin = float((oos['actual_margin'] - oos['predicted_margin']).abs().mean())
    mae_elo_margin = float((oos['actual_margin'] - elo_pred_margin).abs().mean())
    mae_naive_margin = float((oos['actual_margin'] - oos['naive_margin']).abs().mean())
    mae_ml_total = float((oos['actual_total'] - oos['predicted_total']).abs().mean())
    mae_naive_total = float((oos['actual_total'] - oos['naive_total']).abs().mean())
    print(f"\nMAE margin: ML={mae_ml_margin:.2f}  Elo-only={mae_elo_margin:.2f}  naive-pace={mae_naive_margin:.2f}")
    print(f"MAE total:  ML={mae_ml_total:.2f}  naive-pace={mae_naive_total:.2f}")

    # ------------------------------------------------------------------
    # 5. Residual stds & backtest
    # ------------------------------------------------------------------
    hr("5. ENSEMBLE RESIDUAL STDS")
    stds = ensemble.compute_residual_stds(oos, nba_config.ELO_POINTS_PER_MARGIN)
    print(stds)

    hr("6. BACKTEST")
    print("Pricing every bet at Pinnacle's single available snapshot (price_point='Close'). "
          "See this module's docstring for the CLV caveat this implies.")

    ens_cfg = ensemble.EnsembleConfig()

    hr("6a. MIN-EDGE / CONFIDENCE SWEEP")
    sweep_rows = []
    for min_edge in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        for conf_set in [("High",), ("Medium", "High"), ("Low", "Medium", "High")]:
            bt_cfg = backtest.BacktestConfig(
                min_edge_pct=min_edge, allowed_confidence=conf_set, price_point="Close",
            )
            bets = backtest.run_backtest(oos, stds, nba_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
            if bets.empty:
                sweep_rows.append({"min_edge_pct": min_edge, "confidence": "+".join(conf_set),
                                    "bets": 0, "hit_rate": np.nan, "roi_pct": np.nan, "roi_stderr_pct": np.nan})
                continue
            s = backtest.summarize(bets)
            sweep_rows.append({
                "min_edge_pct": min_edge, "confidence": "+".join(conf_set),
                "bets": int(s["bets"].iloc[0]), "hit_rate": float(s["hit_rate"].iloc[0]),
                "roi_pct": float(s["roi_pct"].iloc[0]), "roi_stderr_pct": float(s["roi_stderr_pct"].iloc[0]),
            })
    sweep_df = pd.DataFrame(sweep_rows)
    print(sweep_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Primary config for the detailed breakdown below: same edge/confidence
    # bar the CFB module used (3% min edge, Medium+High confidence), NOT
    # cherry-picked from the sweep above - chosen before looking at the
    # sweep's ROI column, exactly as the CFB build did.
    bt_cfg = backtest.BacktestConfig(
        min_edge_pct=3.0,
        allowed_confidence=("Medium", "High"),
        price_point="Close",
    )
    bets = backtest.run_backtest(oos, stds, nba_config.ELO_POINTS_PER_MARGIN, ens_cfg, bt_cfg)
    print(f"\nPrimary config (edge >= {bt_cfg.min_edge_pct}%, confidence in {bt_cfg.allowed_confidence}): "
          f"{len(bets):,} qualifying bets")

    if bets.empty:
        print("No bets cleared the edge/confidence thresholds - nothing further to summarize.")
        return

    hr("6b. OVERALL SUMMARY")
    print(backtest.summarize(bets).to_string(float_format=lambda x: f"{x:.3f}"))

    hr("6c. SUMMARY BY MARKET")
    print(backtest.summarize(bets, group_cols=["market"]).to_string(float_format=lambda x: f"{x:.3f}"))

    hr("6d. SUMMARY BY MARKET x EDGE BUCKET")
    bets["edge_bucket"] = bets["edge_pct"].apply(backtest.edge_bucket)
    print(backtest.summarize(bets, group_cols=["market", "edge_bucket"]).to_string(float_format=lambda x: f"{x:.3f}"))

    hr("6e. SUMMARY BY SEASON")
    print(backtest.summarize(bets, group_cols=["season"]).to_string(float_format=lambda x: f"{x:.3f}"))

    hr("6f. SUMMARY BY SIDE (within market)")
    print(backtest.summarize(bets, group_cols=["market", "side"]).to_string(float_format=lambda x: f"{x:.3f}"))

    hr("DONE")


if __name__ == "__main__":
    main()
